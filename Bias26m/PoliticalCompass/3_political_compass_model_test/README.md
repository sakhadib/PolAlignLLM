# 🧠 Political Compass Model Test

A robust Python tool for evaluating Large Language Models (LLMs) on political compass questions using the Replicate API. This tool systematically tests models across various political statements and records their responses for bias analysis.

## 📋 Overview

This script processes political compass questions and sends them to LLMs via Replicate's API to gather responses for political bias research. It handles various model schemas, implements retry logic, and provides comprehensive debugging capabilities.

## ✨ Features

- 🔄 **Multi-Schema Compatibility**: Automatically tries multiple input formats to work with different model architectures
- ⚡ **Smart Format Caching**: Remembers successful input formats per model to drastically reduce API calls
- 🛡️ **Robust Error Handling**: Implements exponential backoff retry logic with configurable timeouts
- 📊 **Progress Tracking**: Real-time progress updates with detailed logging
- 🐛 **Comprehensive Debugging**: Verbose mode for troubleshooting model interactions
- 💾 **Resume Capability**: Automatically skips already processed questions
- 🎯 **Consistent Parsing**: Advanced response parsing to extract political alignment codes
- 🔥 **Multiple Runs Support**: Generate N independent predictions with `--n` parameter

## 🚀 Quick Start

### Prerequisites

```bash
pip install replicate pandas python-dotenv

# Optional for accurate token counting (falls back to estimation if not installed)
pip install tiktoken
```

### Environment Setup

Set your Replicate API token:
```bash
# Option 1: Environment variable
export REPLICATE_API_TOKEN="your_token_here"

# Option 2: Alternative variable name
export REPLICATE_KEY="your_token_here"

# Option 3: .env file
echo "REPLICATE_API_TOKEN=your_token_here" > .env
```

### Basic Usage

```bash
# Single run
python modelrun.py --model openai/gpt-4

# Multiple independent runs (10 predictions per question)
python modelrun.py --model openai/gpt-5-nano --n 10
```

### Advanced Usage

```bash
# Multiple runs with custom settings
python modelrun.py \
  --model meta/llama-2-70b-chat \
  --n 5 \
  --csv "/path/to/custom/questions.csv" \
  --verbose \
  --sleep 1.0 \
  --retries 5 \
  --on_fail random
```

## 📖 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **Required** | Replicate model ID (e.g., `owner/model[:version]`) |
| `--n` | int | `1` | Number of independent runs to perform (creates pred_[1]_model, pred_[2]_model, etc.) |
| `--csv` | str | `questions.csv` | Path to CSV file with political questions |
| `--sleep` | float | `0.0` | Seconds to sleep between API calls |
| `--retries` | int | `3` | Number of retry attempts per question |
| `--timeout` | float | `60.0` | Maximum backoff delay in seconds |
| `--verbose` | flag | `False` | Enable detailed debugging output |
| `--on_fail` | str | `na` | Policy for unparseable responses: `na`, `d`, `random`, `reask` |
| `--seed` | int | `0` | Random seed for `--on_fail random` |
| `--error_log` | str | auto | Path to JSONL error log file |

## 📁 Input File Format

Your CSV file must contain these columns:

```csv
question_id,question_text
1,"Economic globalization is beneficial for working people"
2,"The government should regulate business more strictly"
...
```

## 🎯 Response Codes

The script maps model responses to these standardized codes:

| Code | Meaning |
|------|---------|
| `sd` | Strongly Disagree |
| `d` | Disagree |
| `a` | Agree |
| `sa` | Strongly Agree |

## 🔧 How It Works

### 1. **Smart Format Caching**
The script implements an intelligent caching system:
- **First Question**: Tries multiple input formats (discovery mode) and caches the successful one
- **Subsequent Questions**: Uses the cached format directly, reducing API calls by ~99%
- **Fallback**: If cached format fails, automatically rediscovers and updates cache

### 2. **Multi-Schema Input Generation**
The script generates multiple input variants to ensure compatibility:
- Simple prompt format (`prompt`, `temperature: 0.7`, `max_new_tokens`)
- Chat-style messages (`messages` array with system/user roles)
- Alternative parameter names (`input`, `text`, `n_predict`, `max_tokens`)
- With/without token limits and stop sequences

### 3. **Smart Response Parsing**
- Extracts the **last** occurrence of valid codes to handle prompt echoes
- Falls back to keyword matching for verbose responses
- Handles multiple fallback policies for unparseable responses

### 4. **Robust Error Handling**
- Tries multiple input schemas before failing
- Implements exponential backoff for retries
- Gracefully handles streaming responses

## 📊 Output

The script adds new columns to your CSV file for multiple runs:
```
pred_[1]_<sanitized_model_name>
pred_[2]_<sanitized_model_name>
...
pred_[N]_<sanitized_model_name>
```

Example output columns for `--model openai/gpt-5-nano --n 3`:
- `pred_[1]_openai_gpt_5_nano`
- `pred_[2]_openai_gpt_5_nano` 
- `pred_[3]_openai_gpt_5_nano`

**Performance Optimization**: After the first question, the script uses cached input formats, reducing API calls from ~12 per question to just 1 per question.

## 🐛 Debugging

### Enable Verbose Mode
```bash
python modelrun.py --model your-model --verbose
```

### Debug Output Includes:
- ✅ Script initialization steps
- 🔄 Progress through questions
- 🎯 Model call attempts and variants
- 📝 Raw responses vs parsed results
- ⚠️ Detailed error messages

### Sample Debug Output:
```
DEBUG: Script starting...
DEBUG: Model=openai/gpt-5-nano, N=10, Verbose=True, OnFail=na
DEBUG: API key verified
DEBUG: CSV file found at D:\Research\BiasStudyPaper\political_compass_model_test\questions.csv
DEBUG: CSV loaded with 50 rows
DEBUG: Target columns: ['pred_[1]_openai_gpt_5_nano', 'pred_[2]_openai_gpt_5_nano', ...]

=== Run 1/10: Writing into column 'pred_[1]_openai_gpt_5_nano' ===
DEBUG: Existing filled answers: 0/50

[Q1] (1/50)
  DEBUG: Discovering format for openai/gpt-5-nano, trying 12 variants
  DEBUG: Discovery variant #1: ['prompt', 'temperature', 'max_new_tokens']
  DEBUG: Discovery variant #1 produced len=2
  DEBUG: Cached successful format #1 for openai/gpt-5-nano
  DEBUG: Cached template: ['temperature', 'max_new_tokens']
[Q1] -> d

[Q2] (2/50)
  DEBUG: Using cached format for openai/gpt-5-nano: ['temperature', 'max_new_tokens']
  DEBUG: Cached format worked, got len=2
[Q2] -> a
```

## 🔒 Error Recovery

### Common Issues and Solutions:

**API Rate Limits:**
```bash
python modelrun.py --model your-model --sleep 2.0
```

**Model Schema Incompatibility:**
- The script automatically tries 12+ input variants
- Check verbose output to see which variants are failing

**Network Timeouts:**
```bash
python modelrun.py --model your-model --retries 10 --timeout 120
```

**Resume Interrupted Runs:**
- Simply re-run the same command
- Already processed questions are automatically skipped

## 📈 Performance Tips

1. **Leverage Format Caching**: The script automatically caches successful input formats, making subsequent calls ~99% faster
2. **Use appropriate sleep delays** for rate-limited APIs (`--sleep 1.0`)
3. **Enable verbose mode** only for debugging (increases output significantly)
4. **Start with small test files** to validate model compatibility before large runs
5. **Use multiple runs** (`--n 10`) to study response consistency and reliability
6. **Monitor progress** - the script shows real-time status updates per run and question
7. **Configure failure policies** with `--on_fail` for consistent handling of unparseable responses

## 🧪 Example Research Workflow

```bash
# 1. Test a single model with verbose output
python modelrun.py --model openai/gpt-4 --n 1 --verbose

# 2. Run production batch with multiple runs and rate limiting
python modelrun.py --model openai/gpt-5-nano --n 10 --sleep 1.5

# 3. Process multiple models with consistent parameters
for model in "openai/gpt-4" "anthropic/claude-2" "meta/llama-2-70b-chat"; do
    python modelrun.py --model "$model" --n 5 --sleep 2.0 --on_fail random
done

# 4. Resume interrupted runs (automatically skips completed questions)
python modelrun.py --model openai/gpt-5-nano --n 10  # Continues where it left off
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests! Areas for improvement:
- Additional model schema support
- Enhanced response parsing algorithms
- Performance optimizations and caching strategies
- Additional output formats (JSON, JSONL)
- Statistical analysis tools for multiple runs
- Integration with other model APIs

## 📄 License

This tool is part of a political bias research project. Please ensure compliance with your institution's research ethics guidelines and Replicate's terms of service.

---

## Contacts

For inquiries, please reach out to the project maintainers at [sakhadib@gmail.com](mailto:sakhadib@gmail.com).

**Happy Testing! 🚀**

For questions or issues, please check the verbose output first, then consult the debugging section above.
