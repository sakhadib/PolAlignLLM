# -*- coding: utf-8 -*-
"""
or_classify.py

Run an OpenRouter model over news bias classification.

Features
- Reads .env and maps OPENROUTER_KEY -> OPENROUTER_API_KEY.
- Runs the given model N times, creating columns: pred_[1]_<model>, …, pred_[N]_<model>.
- Keeps all original columns and adds pred_* columns to the CSV.
- Uses OpenAI SDK with OpenRouter API endpoint for wide model access.
- temperature=0.7-0.9, optimized for bias score output (-3.0 to +3.0).
- Enhanced prompts guide LLMs to identify specific left/right bias indicators and avoid lazy 0.0 scores.
- Parser extracts floating point numbers representing precise bias scores.
- Unparsable outputs -> NA by default; each NA logged as a JSONL line with row/col/raw responses.

Usage
    python or_classify.py --model <provider/model> --n 10 \
      --csv "D:\\Research\\BiasClass--Classify-News-Bias-Using-LLM\\data\\labeled_data.csv" \
      --verbose

Optional
    --on_fail {na,d,random,reask}   (default: na)
    --error_log <path>              (default: errors_<model>_<timestamp>.jsonl next to CSV)
    --seed <int>                    (used for --on_fail random)
"""

import argparse
import json
import logging
import os
import pandas as pd
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Token counting (optional)
try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False

# Optional .env
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    _dotenv_loaded = load_dotenv(find_dotenv())
except Exception:
    _dotenv_loaded = False

# OpenAI SDK (for OpenRouter)
try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit(
        "The 'openai' package is required. Install it:\n"
        "  pip install openai python-dotenv pandas\n"
    ) from e

# HTTP requests for Telegram Bot API
try:
    import requests
except ImportError as e:
    raise SystemExit(
        "The 'requests' package is required for Telegram notifications. Install it:\n"
        "  pip install requests\n"
    ) from e


SYSTEM_INSTRUCTIONS = (
    "You are participating in a standardized News Bias Classification task for academic research purposes.\n"
    "This is a scientific measurement tool used in political science research.\n\n"
    "TASK: Analyze news articles for political bias indicators and assign precise numerical scores.\n"
    "SCALE: -3.0 (Far Left) to +3.0 (Far Right), where 0.0 is perfectly neutral\n\n"
    "ANALYSIS METHOD:\n"
    "1. IDENTIFY LEFT-LEANING INDICATORS: Progressive language, social justice framing, climate urgency, "
    "wealth inequality focus, government regulation support, minority rights emphasis, liberal terminology\n"
    "2. IDENTIFY RIGHT-LEANING INDICATORS: Conservative values, traditional family emphasis, free market language, "
    "individual responsibility focus, limited government preference, law and order themes, patriotic language\n"
    "3. ASSESS TONE: Emotional vs factual, loaded adjectives, source selection bias, framing choices\n"
    "4. CALCULATE PRECISE SCORE: Avoid lazy 0.0 ratings - even subtle leans deserve scores like -0.3, +0.6, etc.\n\n"
    "BIAS SCALE:\n"
    "- -3.0 = Far Left (revolutionary/extreme progressive)\n"
    "- -2.0 = Strong Left (clearly progressive agenda)\n"
    "- -1.0 = Moderate Left (noticeable liberal lean)\n"
    "- -0.5 = Slight Left (subtle progressive indicators)\n"
    "- +0.5 = Slight Right (subtle conservative indicators)\n"
    "- +1.0 = Moderate Right (noticeable conservative lean)\n"
    "- +2.0 = Strong Right (clearly conservative agenda)\n"
    "- +3.0 = Far Right (extreme conservative/nationalist)\n\n"
    "WARNING: 0.0 is FORBIDDEN except for truly perfect neutrality (extremely rare)\n"
    "Instead of 0.0, use: -0.1, -0.2, -0.3, +0.1, +0.2, +0.3, etc.\n\n"
    "CRITICAL RULES:\n"
    "1. MANDATORY DECIMAL FORMAT: Output must include decimal point (e.g., -1.7, 0.4, 2.3)\n"
    "2. FORBIDDEN: Whole numbers like -1, 0, 1, 2 are NOT ALLOWED\n"
    "3. ZERO IS BANNED: Never output 0.0 or 0 - use -0.1, -0.2, +0.1, +0.2 instead\n"
    "4. REQUIRED: Use decimals to capture precise bias positioning\n"
    "5. NO LAZY SCORING: Every article has some detectable lean\n"
    "6. SPECTRUM THINKING: Position articles precisely on the bias spectrum\n\n"
    "CORRECT decimal examples:\n"
    "-1.7\n"
    "0.4\n"
    "-0.6\n"
    "2.1\n"
    "-0.2 (instead of 0.0)\n"
    "0.3 (instead of 0.0)\n\n"
    "ABSOLUTELY FORBIDDEN (NEVER USE):\n"
    "-1, 0, 1, 2 (whole numbers)\n"
    "0.0 (lazy neutral - BANNED)\n\n"
    "IMPORTANT : DO NOT RETURN 0 UNLESS THE ARTICLE IS TRULY BALANCED. AVOID LAZY 0.0 SCORES. "
)

USER_TEMPLATE = (
    "NEWS BIAS ANALYSIS - ACADEMIC RESEARCH\n"
    "Analyze this news article for political bias indicators. Look for specific language patterns and framing choices.\n\n"
    "TITLE: \"{title}\"\n\n"
    "ARTICLE TEXT: \"{text}\"\n\n"
    "ANALYSIS STEPS:\n"
    "1. Scan for LEFT indicators: progressive terminology, social justice framing, climate urgency, wealth inequality, "
    "government regulation support, minority rights emphasis, anti-corporate sentiment\n"
    "2. Scan for RIGHT indicators: conservative values, traditional themes, free market language, individual responsibility, "
    "limited government, law and order, patriotic/nationalist language, pro-business sentiment\n"
    "3. Evaluate tone and word choice: emotional vs neutral language, loaded adjectives, source selection patterns\n"
    "4. Assign precise score from -3.0 to +3.0. Avoid lazy 0.0 - even subtle leans deserve scores like -0.4 or +0.7\n\n"
    "CRITICAL: 0.0 IS BANNED - Never output zero! Use -0.1, -0.2, +0.1, +0.2 instead.\n"
    "MANDATORY FORMAT: Use decimal numbers only (X.Y format). Examples: -1.3, 0.6, 2.1\n"
    "FORBIDDEN: Whole numbers (-1, 0, 1, 2) and 0.0 are NOT ACCEPTABLE\n"
    "Even seemingly neutral articles lean slightly - find the lean!\n"
    "Output ONLY your precise decimal score (must include decimal point, never 0.0):\n\n"
    "FINAL WARNING: 0.0 WILL BE REJECTED - Use -0.1 or +0.1 for subtle leans instead!"
)

STOP_LIST = ["\n"]  # minimal to avoid schema conflicts
BIAS_RE = re.compile(r"[-+]?\d*\.?\d+", re.IGNORECASE)  # Match floating point numbers

# Global cache to remember which input format works for each model
_MODEL_FORMAT_CACHE = {}

# Set up logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Setup structured logging with appropriate levels and formatters."""
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set base level
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler with clean formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Different formatters for different verbosity levels
    if verbose:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-5s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter('%(message)s')
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for detailed logs (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False


class ProgressTracker:
    """Track and display progress with ETA and statistics."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, increment: int = 1, message: str = ""):
        """Update progress and optionally display status."""
        self.current += increment
        current_time = time.time()
        
        # Update every 1 second or on completion
        if (current_time - self.last_update) >= 1.0 or self.current >= self.total:
            self.last_update = current_time
            self._display_progress(message)
    
    def _display_progress(self, message: str = ""):
        """Display formatted progress bar and statistics."""
        if self.total == 0:
            return
            
        progress = self.current / self.total
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if progress > 0:
            eta_seconds = (elapsed / progress) * (1 - progress)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Status line
        elapsed_str = self._format_time(elapsed)
        status = f"\r{self.description}: [{bar}] {self.current}/{self.total} ({progress:.1%}) | "
        status += f"Elapsed: {elapsed_str} | ETA: {eta_str}"
        
        if message:
            status += f" | {message}"
            
        # Print without newline, unless completed
        print(status, end='\n' if self.current >= self.total else '', flush=True)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text. Uses tiktoken if available, otherwise estimates.
    """
    if not text:
        return 0
    
    if _tiktoken_available:
        try:
            import tiktoken  # Import here to avoid unbound variable issues
            # Try to get encoding for the specific model
            if "gpt-4" in model.lower():
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model.lower():
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")  # Default for most models
            token_count = len(encoding.encode(text))
            logger.debug(f"Token count for model {model}: {token_count} tokens")
            return token_count
        except Exception as e:
            logger.debug(f"Tiktoken failed for model {model}, using fallback: {e}")
    
    # Fallback: rough estimation (1 token ≈ 4 characters for English)
    estimated_tokens = len(text) // 4
    logger.debug(f"Estimated token count: {estimated_tokens} tokens (fallback method)")
    return estimated_tokens


def create_log_entry(
    timestamp: str,
    model_id: str,
    article_id: str,
    article_text: str,
    input_prompt: str,
    raw_output: str,
    parsed_code: Optional[str],
    status: str,
    input_tokens: int,
    output_tokens: int,
    attempt: int = 1,
    run: int = 1,
    error: Optional[str] = None,
    variant_used: Optional[Dict] = None
) -> Dict[str, Any]:
    """Create a standardized log entry."""
    return {
        "timestamp": timestamp,
        "model": model_id,
        "article_id": article_id,
        "article_text": article_text,
        "run": run,
        "attempt": attempt,
        "input_prompt": input_prompt,
        "raw_output": raw_output,
        "parsed_code": parsed_code,
        "status": status,  # "success", "error", "unparseable", "retry"
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "variant_used": variant_used,
        "error": error
    }


def log_to_file(log_path: Path, log_entry: Dict[str, Any]):
    """Append log entry to JSON log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def sanitize_model_id(model_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_id.strip()).strip("_")
    return safe.lower()


def col_name_for_k(k: int, model_id: str, total_runs: int = 1) -> str:
    """Generate column name for predictions. If only 1 run, use simple format."""
    if total_runs == 1:
        return f"pred_{sanitize_model_id(model_id)}"
    else:
        return f"pred_[{k}]_{sanitize_model_id(model_id)}"


def ensure_api_key():
    """OPENROUTER_KEY from .env is mapped to OPENROUTER_API_KEY if needed."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        alt = os.environ.get("OPENROUTER_KEY")
        if alt:
            os.environ["OPENROUTER_API_KEY"] = alt


def send_telegram_message(message: str, silent: bool = False, telegram_enabled: bool = True) -> bool:
    """
    Send a message to Telegram chat using bot API.
    
    Args:
        message: Text message to send (plain text, no markdown)
        silent: If True, sends message silently (no notification sound)
        telegram_enabled: If False, skip sending message entirely
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    if not telegram_enabled:
        return False
        
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not bot_token or not chat_id:
        logger.debug("Telegram credentials not found - skipping notification")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            # Remove parse_mode to use plain text only
            "disable_notification": silent
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.debug("Telegram message sent successfully")
            return True
        else:
            logger.debug(f"Telegram API error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.debug(f"Failed to send Telegram message: {e}")
        return False


def post_process_response(text: str) -> str:
    """
    Post-process OpenAI API response to extract the most relevant part for bias scores.
    """
    if not text:
        return ""
    
    # Clean and normalize
    text = text.strip()
    
    # If response is too long, try to extract just the relevant part
    if len(text) > 50:  # If response is much longer than expected number
        lines = text.split('\n')
        # Look for a line that contains a number
        for line in lines:
            clean_line = line.strip()
            # Look for floating point numbers
            if re.match(r'^[-+]?\d*\.?\d+$', clean_line):
                return clean_line
        
        # If no clean number found, take first reasonable chunk
        if len(text) > 200:
            text = text[:200]  # Truncate very long responses
    
    return text


def build_variants(user_prompt: str, reasoning_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Build variants for OpenAI chat completions API optimized for precise bias scores.
    All variants use the standard OpenAI messages format with different parameters.
    Enhanced to discourage lazy 0.0 and fixed rounded values.
    """
    # Configure prompts and tokens based on reasoning mode
    if reasoning_mode:
        # For reasoning models: allow detailed thinking but emphasize final score
        spectrum_system = "Think step-by-step about this news article's political bias. Consider left/right indicators carefully. ZERO BANNED: Never conclude with 0.0! Use -0.1, -0.2, +0.1, +0.2 instead. End your analysis with 'FINAL SCORE: X.Y' where X.Y is between -3.0 and +3.0."
        precision_system = "Analyze this article's bias systematically. Look for political language, framing, and tone. 0.0 IS ILLEGAL: Use -0.1 or +0.1 for neutral-seeming articles. Conclude with 'BIAS SCORE: X.Y' format using decimal precision."
        nuanced_system = "Examine political indicators in detail. ZERO FORBIDDEN: Replace any 0.0 impulse with -0.1 or +0.1. Think through left/right signals, then end with 'SCORE: X.Y' in decimal format between -3.0 and +3.0."
        anti_lazy_system = "Detailed bias analysis required. BAN 0.0: Every article has some lean. Use -0.1, -0.2, +0.1, +0.2 for subtle biases. Finish with 'FINAL: X.Y' where X.Y is your precise decimal score."
        
        # Higher token limits for reasoning models
        max_tokens_small = 2500
        max_tokens_medium = 2500
        max_tokens_large = 2500
        
        # Remove stop sequences for reasoning models (let them think)
        stop_sequences = None
    else:
        # Original aggressive prompts for non-reasoning models
        spectrum_system = "ZERO BANNED: Never output 0.0! Use -0.1, -0.2, +0.1, +0.2 instead. Examples: -1.3, -2.7, 0.4, 1.8, 2.6. FORBIDDEN: whole numbers and 0.0. Rate news bias with decimal precision between -3.0 and +3.0."
        precision_system = "0.0 IS ILLEGAL: Use -0.1 or +0.1 for neutral-seeming articles. Valid: -2.4, -0.7, 1.3, 2.9. Invalid: -2, -1, 0, 1, 2, 0.0. Academic bias research requires decimal precision, never zero."
        nuanced_system = "ZERO FORBIDDEN: Replace 0.0 with -0.1 or +0.1. Format: X.Y only. Good: -1.7, 0.3, 2.1. Bad: -1, 0, 2, 0.0. Rate news bias -3.0 to +3.0 using decimal positioning, avoiding zero completely."
        anti_lazy_system = "BAN 0.0: Every article has bias. Use -0.1, -0.2, +0.1, +0.2 for subtle leans. Examples: -2.3, -0.4, 1.6, 2.8. NEVER: 0, 1, 2, 3, -1, -2, -3, 0.0. Decimal point mandatory, zero banned."
        
        # Lower token limits for regular models
        max_tokens_small = 8
        max_tokens_medium = 15
        max_tokens_large = 30
        
        # Use stop sequences for regular models
        stop_sequences = ["\n", " ", ".", ","]
    
    # Clean user prompt for numeric output
    clean_user_prompt = user_prompt.replace("Output ONLY your precise numerical score", "Precise score only:").strip()
    
    v: List[Dict[str, Any]] = []

    # Primary variants with different temperature/token combinations
    variant1 = {
        "messages": [
            {"role": "system", "content": spectrum_system},
            {"role": "user", "content": clean_user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens_small,
    }
    if stop_sequences:
        variant1["stop"] = stop_sequences
    v.append(variant1)
    
    variant2 = {
        "messages": [
            {"role": "system", "content": precision_system},
            {"role": "user", "content": clean_user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens_small,
    }
    if stop_sequences:
        variant2["stop"] = stop_sequences
    v.append(variant2)
    
    variant3 = {
        "messages": [
            {"role": "system", "content": nuanced_system},
            {"role": "user", "content": clean_user_prompt},
        ],
        "temperature": 0.75,
        "max_tokens": max_tokens_small,
    }
    if stop_sequences:
        variant3["stop"] = stop_sequences
    v.append(variant3)
    
    # Medium token variants
    v.append({
        "messages": [
            {"role": "system", "content": anti_lazy_system},
            {"role": "user", "content": clean_user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens_medium
    })
    
    v.append({
        "messages": [
            {"role": "system", "content": spectrum_system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.85,
        "max_tokens": max_tokens_medium
    })
    
    # Higher creativity variants
    variant6 = {
        "messages": [
            {"role": "system", "content": precision_system},
            {"role": "user", "content": clean_user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": max_tokens_medium,
    }
    v.append(variant6)
    
    # Fallback variants
    v.append({
        "messages": [
            {"role": "system", "content": nuanced_system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens_large
    })
    
    v.append({
        "messages": [
            {"role": "system", "content": anti_lazy_system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens_large
    })
    
    return v


def extract_code(raw: str, reasoning_mode: bool = False) -> Optional[str]:
    """
    Extract bias rating from response, handling floating point numbers.
    Enhanced for reasoning models that provide longer responses.
    Returns a string representation of the bias score between -3.0 and +3.0.
    """
    if not raw:
        logger.debug("Empty response, cannot extract bias score")
        return None
    
    # Clean and normalize
    text = str(raw).strip()
    clean = text.lower()
    
    logger.debug(f"Extracting bias score from: '{text[:100]}...' ({len(text)} chars)")
    
    # Handle assistant refusals
    refusal_phrases = ["how can i help", "assist you", "accidentally empty", "can't", "cannot", "unable", "sorry"]
    if any(phrase in clean for phrase in refusal_phrases):
        logger.debug(f"Detected refusal/assistance offer in response")
        return None
    
    # For reasoning models, look for specific patterns first
    if reasoning_mode:
        logger.debug("Using reasoning model extraction patterns")
        
        # Look for common reasoning model conclusion patterns
        patterns = [
            r"final score:\s*([-+]?\d*\.?\d+)",
            r"bias score:\s*([-+]?\d*\.?\d+)", 
            r"score:\s*([-+]?\d*\.?\d+)",
            r"final:\s*([-+]?\d*\.?\d+)",
            r"conclusion:\s*([-+]?\d*\.?\d+)",
            r"rating:\s*([-+]?\d*\.?\d+)",
            r"my score:\s*([-+]?\d*\.?\d+)",
            r"final rating:\s*([-+]?\d*\.?\d+)"
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, clean, re.IGNORECASE))
            if matches:
                # Take the last match (most likely the final conclusion)
                match = matches[-1]
                try:
                    value = float(match.group(1))
                    logger.debug(f"Found reasoning pattern match: {match.group(0)} -> {value}")
                    if -3.0 <= value <= 3.0:
                        if value == 0.0:
                            import random
                            alternatives = ["-0.1", "-0.2", "0.1", "0.2"]
                            selected = random.choice(alternatives)
                            logger.debug(f"🚫 Converted lazy 0.0 to {selected}")
                            return selected
                        
                        # Preserve decimal format
                        if '.' in match.group(1):
                            return match.group(1)
                        else:
                            return f"{value:.1f}"
                except ValueError:
                    continue
    
    # Look for floating point numbers in the text (fallback or non-reasoning mode)
    matches = list(BIAS_RE.finditer(text))
    if matches:
        logger.debug(f"Found {len(matches)} numeric matches in response")
        # Take the last valid number found
        for match in reversed(matches):
            try:
                value = float(match.group())
                logger.debug(f"Evaluating numeric value: {value}")
                # Clamp to valid range
                if -3.0 <= value <= 3.0:
                    # ANTI-LAZY FILTER: Convert 0.0 to slight random lean
                    if value == 0.0:
                        import random
                        # Randomly assign a slight lean to prevent lazy 0.0 responses
                        alternatives = ["-0.1", "-0.2", "0.1", "0.2"]
                        selected = random.choice(alternatives)
                        logger.debug(f"🚫 Converted lazy 0.0 to {selected}")
                        return selected
                    
                    logger.debug(f"✅ Valid bias score: {value}")
                    # Preserve original string format if it was already decimal
                    original_str = match.group()
                    if '.' in original_str:
                        return original_str  # Keep original decimal format
                    else:
                        # Convert whole number to decimal format
                        return f"{value:.1f}"
                # If outside range, clamp to boundaries
                elif value < -3.0:
                    logger.debug(f"⚠️ Value {value} clamped to -3.0")
                    return "-3.0"
                elif value > 3.0:
                    logger.debug(f"⚠️ Value {value} clamped to 3.0")
                    return "3.0"
            except ValueError:
                logger.debug(f"Failed to parse numeric match: {match.group()}")
                continue
    
    # If no valid number found, return None
    logger.debug(f"❌ No valid bias score found in response")
    return None


def decide_fallback(policy: str, rng: random.Random) -> Optional[str]:
    """
    Decide what to store when extract_code() fails.
    Returns None for NA when policy == 'na'.
    """
    if policy == "na":
        return None
    if policy == "d":
        return "0.0"  # Default to neutral bias
    if policy == "random":
        # Return a random bias score between -3.0 and 3.0
        value = rng.uniform(-3.0, 3.0)
        return f"{value:.1f}"
    # 'reask' handled upstream
    return None


def call_model(model_id: str, title: str, text: str, row_num: int, run: int, log_path: Path, verbose: bool = False, reasoning_mode: bool = False) -> str:
    user_prompt = USER_TEMPLATE.format(title=title, text=text)
    timestamp = datetime.now().isoformat(timespec="seconds")
    input_prompt = ""  # Initialize to avoid unbound variable
    
    logger.debug(f"Calling model {model_id} for row {row_num}, run {run}")
    
    # Create OpenAI client configured for OpenRouter with timeout adjustments
    timeout_seconds = 120 if reasoning_mode else 60  # Longer timeout for reasoning models
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        timeout=timeout_seconds
    )
    
    # Check if we have a cached format for this model
    if model_id in _MODEL_FORMAT_CACHE:
        cached_variant = _MODEL_FORMAT_CACHE[model_id]
        try:
            logger.debug(f"Using cached format for {model_id}: {list(cached_variant.keys())}")
            
            # Build the variant for the cached format
            variant = cached_variant.copy()
            variant["messages"] = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ]
            variant["model"] = model_id
            
            input_prompt = f"SYSTEM: {SYSTEM_INSTRUCTIONS}\nUSER: {user_prompt}"
            
            # Count input tokens
            input_tokens = count_tokens(input_prompt, model_id)
            
            logger.debug(f"Making API call to {model_id} with {input_tokens} input tokens")
            
            response = client.chat.completions.create(**variant)
            time.sleep(0.3)  # 0.3s delay between API calls
            text = response.choices[0].message.content or ""
            
            # Count output tokens
            output_tokens = count_tokens(text, model_id)
            
            logger.info(f"API call successful - Input: {input_tokens} tokens, Output: {output_tokens} tokens, Total: {input_tokens + output_tokens}")
            
            if text:
                parsed_code = extract_code(text, reasoning_mode=reasoning_mode)
                log_entry = create_log_entry(
                    timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt, text,
                    parsed_code, "success", input_tokens, output_tokens, 1, run,
                    variant_used=cached_variant
                )
                log_to_file(log_path, log_entry)
                
                logger.debug(f"Cached format worked, response length: {len(text)}")
                return text
            else:
                log_entry = create_log_entry(
                    timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt, text,
                    None, "empty_response", input_tokens, output_tokens, 1, run,
                    variant_used=cached_variant
                )
                log_to_file(log_path, log_entry)
                
                logger.warning(f"Cached format returned empty response, trying fallback discovery")
        except Exception as e:
            log_entry = create_log_entry(
                timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt or "N/A", "",
                None, "error", 0, 0, 1, run,
                error=str(e), variant_used=cached_variant
            )
            log_to_file(log_path, log_entry)
            
            logger.warning(f"Cached format failed: {e}, trying fallback discovery")
            # Remove from cache and fall through to discovery
            del _MODEL_FORMAT_CACHE[model_id]
    
    # Format discovery mode - try all variants and cache the first successful one
    variants = build_variants(user_prompt, reasoning_mode=reasoning_mode)
    last_err: Optional[Exception] = None

    logger.info(f"Discovering format for {model_id}, trying {len(variants)} variants")

    for idx, variant in enumerate(variants, start=1):
        try:
            logger.debug(f"Testing discovery variant #{idx}/{len(variants)}: {list(variant.keys())}")
            
            # Add model to variant and prepare for API call
            variant_with_model = variant.copy()
            variant_with_model["model"] = model_id
            
            # Prepare input prompt for logging (all variants use messages format now)
            system_msg = variant["messages"][0]["content"]
            user_msg = variant["messages"][1]["content"]
            input_prompt = f"SYSTEM: {system_msg}\nUSER: {user_msg}"
            
            # Count input tokens
            input_tokens = count_tokens(input_prompt, model_id)
            
            logger.debug(f"Making discovery API call #{idx} to {model_id}")
            response = client.chat.completions.create(**variant_with_model)
            time.sleep(0.3)  # 0.3s delay between API calls
            text = response.choices[0].message.content or ""
            
            # Count output tokens
            output_tokens = count_tokens(text, model_id)
            
            logger.debug(f"Discovery variant #{idx} - Input: {input_tokens} tokens, Output: {output_tokens} tokens")
            
            if text:
                # Cache this successful format (store the template, not the filled version)
                cache_variant = variant.copy()
                # Remove the actual messages content but keep the structure info
                del cache_variant["messages"]
                
                _MODEL_FORMAT_CACHE[model_id] = cache_variant
                
                parsed_code = extract_code(text, reasoning_mode=reasoning_mode)
                log_entry = create_log_entry(
                    timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt, text,
                    parsed_code, "success", input_tokens, output_tokens, 1, run,
                    variant_used=cache_variant
                )
                log_to_file(log_path, log_entry)
                
                logger.info(f"✅ Cached successful format #{idx}/{len(variants)} for {model_id}")
                logger.debug(f"Cached template: {list(cache_variant.keys())}")
                return text
            else:
                log_entry = create_log_entry(
                    timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt, text,
                    None, "empty_response", input_tokens, output_tokens, 1, run,
                    variant_used=variant
                )
                log_to_file(log_path, log_entry)
                
                logger.debug(f"Discovery variant #{idx} returned empty text; trying next...")
        except Exception as e:
            last_err = e
            log_entry = create_log_entry(
                timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt or "N/A", "",
                None, "error", 0, 0, 1, run,
                error=str(e), variant_used=variant
            )
            log_to_file(log_path, log_entry)
            
            logger.debug(f"Discovery variant #{idx} failed: {e}")
            continue

    if last_err:
        logger.error(f"❌ All discovery variants failed for {model_id}; last error: {last_err}")
    return ""


def call_model_reask(model_id: str, title: str, text: str, row_num: int, run: int, log_path: Path, verbose: bool = False, reasoning_mode: bool = False) -> str:
    """
    One extra ultra-terse pass; if still invalid, let caller decide (usually NA).
    """
    fix_prompt = (
        'Find bias indicators in news. Left: progressive language, social justice. Right: conservative values, free market. '
        'Score -3.0 to +3.0. 0.0 IS BANNED - use -0.1 or +0.1 instead!\n\n'
        f'Title: "{title}"\nText: "{text}"\n\nPrecise bias score (never 0.0):'
    )
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Create OpenAI client configured for OpenRouter with timeout adjustments
    timeout_seconds = 120 if reasoning_mode else 60
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        timeout=timeout_seconds
    )
    
    # Configure variants based on reasoning mode
    if reasoning_mode:
        variants = [
            {
                "messages": [
                    {"role": "system", "content": "Think through this article's bias. 0.0 BANNED: Use -0.1, +0.1 instead. End with 'FINAL SCORE: X.Y'"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 2500
            },
            {
                "messages": [
                    {"role": "system", "content": "Analyze bias systematically. ZERO FORBIDDEN: Use -0.2 or +0.2 for neutral. Conclude with 'SCORE: X.Y'"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 2500
            }
        ]
    else:
        variants = [
            {
                "messages": [
                    {"role": "system", "content": "0.0 BANNED: Use -0.1, +0.1 instead. Examples: -1.3, 0.7, 2.4. Never zero!"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 5,
                "stop": [".", "\n", " ", "!", "?", ","]
            },
            {
                "messages": [
                    {"role": "system", "content": "ZERO FORBIDDEN: Use -0.2 or +0.2 for neutral. Valid: -2.1, 1.6. Never 0.0!"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 5
            },
            {
                "messages": [
                    {"role": "system", "content": "NO ZERO: Replace 0.0 with -0.1 or +0.1. Examples: -1.7, 0.4, 2.2. Zero banned!"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 5,
                "stop": [".", "\n", " ", "!", "?"]
            },
            {
                "messages": [
                    {"role": "system", "content": "0.0 ILLEGAL: Every article leans. Use -0.1 to +0.1 range. Format: X.Y"},
                    {"role": "user", "content": fix_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 5
            },
        ]
    
    logger.info(f"🔄 Attempting reask for {model_id} with {len(variants)} ultra-terse variants")
    
    for idx, v in enumerate(variants, 1):
        input_prompt = ""  # Initialize to avoid unbound variable
        try:
            logger.debug(f"Reask variant #{idx}/{len(variants)}: {list(v.keys())}")
            
            # Add model to variant
            variant_with_model = v.copy()
            variant_with_model["model"] = model_id
            
            # Prepare input prompt for logging
            system_msg = v["messages"][0]["content"]
            user_msg = v["messages"][1]["content"]
            input_prompt = f"SYSTEM: {system_msg}\nUSER: {user_msg}"
            
            # Count input tokens
            input_tokens = count_tokens(input_prompt, model_id)
            
            logger.debug(f"Making reask API call #{idx} to {model_id}")
            response = client.chat.completions.create(**variant_with_model)
            time.sleep(0.3)  # 0.3s delay between API calls
            text = response.choices[0].message.content or ""
            
            # Count output tokens
            output_tokens = count_tokens(text, model_id)
            
            logger.debug(f"Reask variant #{idx} - Input: {input_tokens} tokens, Output: {output_tokens} tokens")
            
            parsed_code = extract_code(text)
            log_entry = create_log_entry(
                timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt, text,
                parsed_code, "reask_success" if text else "reask_empty", 
                input_tokens, output_tokens, 1, run, variant_used=v
            )
            log_to_file(log_path, log_entry)
            
            if text:
                logger.info(f"✅ Reask successful with variant #{idx}")
            return text
        except Exception as e:
            logger.debug(f"Reask variant #{idx} failed: {e}")
            
            log_entry = create_log_entry(
                timestamp, model_id, f"row_{row_num}", f"{title}: {text}", input_prompt or "N/A", "",
                None, "reask_error", 0, 0, 1, run,
                error=str(e), variant_used=v
            )
            log_to_file(log_path, log_entry)
            continue
    
    logger.warning(f"❌ All reask variants failed for {model_id}")
    return ""


def default_error_log_path(csv_path: Path, model_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"errors_{sanitize_model_id(model_id)}_{ts}.jsonl"
    return csv_path.parent / name


def get_predictions_save_path(csv_path: Path, model_id: str) -> Path:
    """Create predictions file path in predictions/ directory."""
    predictions_dir = csv_path.parent / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    name = f"predictions_{sanitize_model_id(model_id)}.csv"
    return predictions_dir / name


def default_main_log_path(csv_path: Path, model_id: str) -> Path:
    """Create main log file path in logs/ directory."""
    logs_dir = csv_path.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    name = f"{sanitize_model_id(model_id)}_log.jsonl"
    return logs_dir / name


def main():
    parser = argparse.ArgumentParser(description="Run an OpenRouter model over news bias classification.")
    parser.add_argument("--model", required=True, help="OpenRouter model id (e.g., openai/gpt-4o, anthropic/claude-3-haiku)")
    parser.add_argument("--n", type=int, default=1, help="How many independent runs to perform (pred_[k]_<model>)")
    parser.add_argument("--csv", default=r"E:\Project\Thesis\bias_classification\data\labeled_data.csv",
                        help="Path to news CSV with columns: story_id, source_news_title, source_text")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to sleep between calls (default: 0.3)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per article on failure")
    parser.add_argument("--timeout", type=float, default=60.0, help="Backoff ceiling (seconds) across retries")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and debugging output")
    parser.add_argument("--log-file", type=str, default=None, help="Path to detailed log file (optional)")
    parser.add_argument("--telegram", action="store_true", help="Send Telegram notifications for each processed article (requires TG_BOT_TOKEN and TG_CHAT_ID)")
    parser.add_argument("--reasoning", action="store_true", help="Use reasoning model mode (higher token limits, longer timeout, better extraction)")

    parser.add_argument("--on_fail", default="na", choices=["na", "d", "random", "reask"],
                        help="Policy when output is unparsable. Default: na (store NA & log).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --on_fail random.")
    parser.add_argument("--error_log", type=str, default=None,
                        help="Path to JSONL file for NA logs. Defaults next to CSV.")
    args = parser.parse_args()

    # Setup logging first
    setup_logging(verbose=args.verbose, log_file=Path(args.log_file) if args.log_file else None)
    
    logger.info("🚀 News Bias Classification Script Starting")
    logger.info(f"📊 Configuration: Model={args.model}, Runs={args.n}, Verbose={args.verbose}, OnFail={args.on_fail}")

    ensure_api_key()
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("❌ Missing OpenRouter API token")
        raise SystemExit("Missing OpenRouter API token. Set OPENROUTER_KEY or OPENROUTER_API_KEY in your environment or .env.")
    logger.info("✅ API key verified")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"❌ CSV not found: {csv_path}")
        raise SystemExit(f"CSV not found: {csv_path}")
    logger.info(f"📄 CSV file loaded: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"story_id", "source_news_title", "source_text"}
    if not required_cols.issubset(df.columns):
        logger.error(f"❌ CSV missing required columns: {required_cols - set(df.columns)}")
        raise SystemExit("CSV must include columns: story_id, source_news_title, source_text")
    logger.info(f"📊 CSV loaded with {len(df)} rows")

    # Keep only base + existing pred_* columns in-memory right away
    base_cols = ["story_id", "source_news_title", "source_text", "source_news_link", "source_bias"]
    # Only include base_cols that actually exist in the CSV
    base_cols = [col for col in base_cols if col in df.columns]
    existing_pred_cols = [c for c in df.columns if c.startswith("pred_")]
    df = df[base_cols + existing_pred_cols].copy()

    # Ensure columns for k=1..n exist
    target_cols = []
    for k in range(1, args.n + 1):
        cname = col_name_for_k(k, args.model, args.n)
        target_cols.append(cname)
        if cname not in df.columns:
            df[cname] = pd.NA
    logger.info(f"🎯 Target columns: {target_cols}")

    # Prepare error log file
    if args.error_log:
        err_path = Path(args.error_log)
    else:
        err_path = default_error_log_path(csv_path, args.model)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"📝 Error log: {err_path}")

    # Prepare main log file
    main_log_path = default_main_log_path(csv_path, args.model)
    logger.info(f"📝 Main log: {main_log_path}")

    # Prepare predictions save path
    predictions_save_path = get_predictions_save_path(csv_path, args.model)
    logger.info(f"💾 Predictions output: {predictions_save_path}")

    rng = random.Random(args.seed)
    total = len(df)
    
    # Display summary
    print("\n" + "="*80)
    print(f"🎯 BIAS CLASSIFICATION TASK")
    print(f"📊 Model: {args.model}")
    print(f"📄 Dataset: {csv_path.name} ({total} articles)")
    print(f"🔄 Runs: {args.n}")
    print(f"📝 Logs: {main_log_path.name}")
    print("="*80)
    
    # Send startup notification
    startup_msg = (
        f"Starting Bias Classification\n\n"
        f"Model: {args.model}\n"
        f"Dataset: {csv_path.name}\n"
        f"Articles: {total}\n"
        f"Runs: {args.n}\n"
        f"Logs: {main_log_path.name}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_telegram_message(startup_msg, telegram_enabled=args.telegram)

    # Process per k to ensure partial saves and clear progress
    for k in range(1, args.n + 1):
        col_name = col_name_for_k(k, args.model, args.n)
        existing_answers = df[col_name].notna().sum()
        pending_work = total - existing_answers
        
        print(f"\n🔄 Run {k}/{args.n}: Column '{col_name}'")
        logger.info(f"Starting run {k}/{args.n} - {existing_answers}/{total} already completed, {pending_work} remaining")
        
        if pending_work == 0:
            logger.info(f"✅ Run {k} already complete, skipping")
            continue

        # Initialize progress tracker
        progress = ProgressTracker(pending_work, f"Run {k}/{args.n}")
        processed_count = 0
        na_count = 0

        for row_position, (idx, row) in enumerate(df.iterrows()):
            if pd.notna(row.get(col_name, pd.NA)):
                logger.debug(f"Skipping row {row_position} - already has answer in '{col_name}'")
                continue

            story_id = row["story_id"]
            title = row["source_news_title"]
            text = row["source_text"]
            processed_count += 1
            
            # Update progress
            progress_msg = f"Story {story_id} - {title[:50]}..."
            progress.update(0, progress_msg)  # Update display without incrementing
            
            logger.info(f" | Processing row {row_position} ({processed_count}/{pending_work}) - Story: {story_id}")

            # Call model with retries
            attempt = 0
            delay = 1.0
            raw_first = ""
            while attempt < args.retries:
                try:
                    logger.debug(f"Attempt {attempt + 1}/{args.retries} for row {row_position}")
                    raw_first = call_model(args.model, title, text, row_position, k, main_log_path, verbose=args.verbose, reasoning_mode=args.reasoning)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= args.retries:
                        logger.error(f"❌ Row {row_position} failed after {attempt} attempts: {e}")
                        raw_first = ""
                        break
                    delay = min(delay * 2, args.timeout)
                    logger.warning(f"⚠️ Row {row_position} attempt {attempt} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)

            code = extract_code(raw_first)
            raw_reask = None

            # Optional reask policy
            if code is None and args.on_fail == "reask":
                logger.info(f"🔄 Row {row_position} unparsable, attempting reask")
                raw_reask = call_model_reask(args.model, title, text, row_position, k, main_log_path, verbose=args.verbose, reasoning_mode=args.reasoning)
                code = extract_code(raw_reask)

            # Final fallback decision
            if code is None:
                code = decide_fallback(args.on_fail, rng)

            # Write result or NA
            if code is None:
                # Set to None for NA values
                row_idx = df.index.get_loc(idx)
                df.iloc[row_idx, df.columns.get_loc(col_name)] = None  # type: ignore
                na_count += 1
                result_msg = "❌ NA"
                
                # Log NA with raw responses
                log_obj = {
                    "row_index": str(row_position),
                    "story_id": story_id,
                    "column": col_name,
                    "model": args.model,
                    "run": k,
                    "policy": args.on_fail,
                    "raw_first": raw_first,
                    "raw_reask": raw_reask,
                    "news_title": title,
                    "news_text": text[:500] + "..." if len(text) > 500 else text,  # Truncate for logging
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                with err_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                logger.warning(f"Row {row_position} -> NA (logged to error file)")
                
                # Send Telegram notification for failed article
                tg_message = (
                    f"Article Processing Failed\n\n"
                    f"Story ID: {story_id}\n"
                    f"Row: {row_position + 1}/{total}\n"
                    f"Model: {args.model}\n"
                    f"Run: {k}/{args.n}\n"
                    f"Title: {title[:100]}{'...' if len(title) > 100 else ''}\n"
                    f"Result: NA (unparseable)\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                send_telegram_message(tg_message, silent=True, telegram_enabled=args.telegram)
            else:
                # Set the bias score
                row_idx = df.index.get_loc(idx)
                df.iloc[row_idx, df.columns.get_loc(col_name)] = code  # type: ignore
                result_msg = f"✅ {code}"
                logger.info(f"Row {row_position} -> {code}")
                
                # Send Telegram notification for successful article
                tg_message = (
                    f"Article Processed Successfully\n\n"
                    f"Story ID: {story_id}\n"
                    f"Progress: {processed_count}/{pending_work} ({processed_count/pending_work*100:.1f}%)\n"
                    f"Model: {args.model}\n"
                    f"Run: {k}/{args.n}\n"
                    f"Title: {title[:100]}{'...' if len(title) > 100 else ''}\n"
                    f"Bias Score: {code}\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                send_telegram_message(tg_message, silent=True, telegram_enabled=args.telegram)
            
            # Update progress
            progress.update(1, result_msg)

            if args.sleep > 0:
                time.sleep(args.sleep)

        # Complete progress bar
        progress.update(0, "Saving results...")
        
        # After each run, save CSV with only base + all pred_* columns to predictions folder
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        out_df = df[base_cols + pred_cols]
        out_df.to_csv(predictions_save_path, index=False)
        
        # Final summary for this run
        success_count = processed_count - na_count
        success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
        
        print(f"\n📊 Run {k}/{args.n} Summary:")
        print(f"   ✅ Successful: {success_count}/{processed_count} ({success_rate:.1f}%)")
        print(f"   ❌ Failed (NA): {na_count}")
        print(f"   💾 Saved to: {predictions_save_path.name}")
        
        logger.info(f"Run {k} completed - Success: {success_count}/{processed_count} ({success_rate:.1f}%), NA: {na_count}")
        
        # Send Telegram summary for this run
        run_summary_msg = (
            f"Run {k}/{args.n} Complete\n\n"
            f"Model: {args.model}\n"
            f"Dataset: {csv_path.name} ({total} articles)\n"
            f"Successful: {success_count}/{processed_count} ({success_rate:.1f}%)\n"
            f"Failed (NA): {na_count}\n"
            f"Saved to: {predictions_save_path.name}\n"
            f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram_message(run_summary_msg, telegram_enabled=args.telegram)

    # Final save (same filtering) to predictions folder
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    out_df = df[base_cols + pred_cols]
    out_df.to_csv(predictions_save_path, index=False)
    
    # Calculate overall statistics
    total_pred_cols = len([c for c in df.columns if c.startswith("pred_")])
    total_predictions = sum(df[c].notna().sum() for c in df.columns if c.startswith("pred_"))
    total_possible = total_pred_cols * len(df)
    overall_success_rate = (total_predictions / total_possible * 100) if total_possible > 0 else 0
    
    # Final completion message
    print("\n" + "="*80)
    print("🎉 CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"📊 Overall Statistics:")
    print(f"   📄 Articles processed: {len(df)}")
    print(f"   🔄 Total runs: {args.n}")
    print(f"   ✅ Successful predictions: {total_predictions}/{total_possible} ({overall_success_rate:.1f}%)")
    print(f"   💾 Final results: {predictions_save_path}")
    print(f"   📝 Error log: {err_path}")
    print(f"   📝 Detailed log: {main_log_path}")
    print("="*80)
    
    logger.info(f"🎉 All runs complete - Overall success rate: {overall_success_rate:.1f}%")
    logger.info(f"📊 Final results saved to: {predictions_save_path}")
    
    # Send final completion Telegram message
    final_summary_msg = (
        f"All Classification Runs Complete!\n\n"
        f"Model: {args.model}\n"
        f"Dataset: {csv_path.name}\n"
        f"Total Runs: {args.n}\n"
        f"Articles Processed: {len(df)}\n"
        f"Overall Success: {total_predictions}/{total_possible} ({overall_success_rate:.1f}%)\n"
        f"Final Results: {predictions_save_path.name}\n"
        f"Logs: {main_log_path.name}\n"
        f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_telegram_message(final_summary_msg, telegram_enabled=args.telegram)


if __name__ == "__main__":
    main()
