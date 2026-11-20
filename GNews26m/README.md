# 🗞️ News Bias Classification with LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust, enterprise-grade system for classifying political bias in news articles using Large Language Models via the Replicate API. This tool provides precise bias scoring with comprehensive logging, error handling, and analytical depth.

## 🚀 Features

### Core Capabilities
- **🎯 Analytical Bias Detection**: Advanced prompt engineering guides LLMs to identify specific left/right bias indicators
- **📊 Continuous Scoring Scale**: Precise -3.0 to +3.0 bias scoring with decimal precision
- **🔄 Multi-Run Reliability**: Supports multiple independent runs for statistical analysis
- **📝 Comprehensive Logging**: Structured logging with progress tracking and detailed audit trails
- **🛡️ Robust Error Handling**: Advanced retry mechanisms with exponential backoff
- **⚡ Smart Caching**: Model format discovery and caching for optimal performance
- **🔍 Row-Based Processing**: Uses row numbers as primary keys for reliable tracking

### Advanced Features
- **🧠 Intelligent Prompt Engineering**: Discourages lazy 0.0 scores, encourages nuanced analysis
- **🔧 Schema-Agnostic Model Support**: Supports various Replicate model input formats
- **📈 Real-Time Progress Tracking**: Visual progress bars with ETA estimation
- **💾 Incremental Saving**: Saves progress after each run to prevent data loss
- **🎛️ Flexible Failure Policies**: Multiple strategies for handling unparseable outputs
- **📊 Token Usage Tracking**: Monitors API costs with detailed token counting

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Data Format](#-data-format)
4. [Usage Guide](#-usage-guide)
5. [Command Reference](#-command-reference)
6. [Bias Scoring System](#-bias-scoring-system)
7. [Output Structure](#-output-structure)
8. [Configuration](#-configuration)
9. [Troubleshooting](#-troubleshooting)
10. [Best Practices](#-best-practices)
11. [API Reference](#-api-reference)

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Replicate API account and token
- Minimum 4GB RAM for large datasets

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/sakhadib/BiasClass--Classify-News-Bias-Using-LLM.git
cd BiasClass--Classify-News-Bias-Using-LLM

# Install required packages
pip install -r requirements.txt

# Or install manually
pip install replicate pandas python-dotenv tiktoken
```

### API Key Setup

**Option 1: Environment Variable**
```bash
export REPLICATE_API_TOKEN="your-api-key-here"
```

**Option 2: .env File (Recommended)**
```bash
echo "REPLICATE_KEY=your-api-key-here" > .env
```

**Option 3: Windows PowerShell**
```powershell
$env:REPLICATE_API_TOKEN="your-api-key-here"
```

## 🚀 Quick Start

### Basic Usage
```bash
cd 2_classify
python classify.py --model "meta/llama-2-70b-chat" --n 1 --verbose
```

### Multiple Runs for Reliability Analysis
```bash
python classify.py --model "meta/llama-2-70b-chat" --n 5 --verbose
```

### Custom Dataset
```bash
python classify.py --model "anthropic/claude-3-sonnet" --csv "path/to/your/data.csv" --n 3
```

## 📊 Data Format

### Required CSV Columns
Your input CSV must contain these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `story_id` | string | Story identifier (may repeat for different sources) | "story_12345" |
| `source_news_title` | string | Article headline/title | "Supreme Court Rules on Healthcare" |
| `source_text` | string | Full article content | "The Supreme Court today..." |

### Optional Columns
These columns will be preserved if present:
- `source_news_link`: URL to original article
- `source_bias`: Known/labeled bias score
- Any existing `pred_*` columns from previous runs

### Data Requirements
- **Row-based tracking**: Script uses row numbers (0, 1, 2, ...) as primary keys
- **UTF-8 encoding**: Ensure proper character encoding for international content
- **No empty required fields**: All three required columns must have valid content

## 📖 Usage Guide

### Basic Classification

```bash
# Single run with detailed output
python classify.py \
    --model "meta/llama-2-70b-chat" \
    --csv "../data/labeled_data.csv" \
    --n 1 \
    --verbose

# Multiple runs for reliability testing
python classify.py \
    --model "mistralai/mixtral-8x7b-instruct-v0.1" \
    --n 10 \
    --sleep 0.5 \
    --verbose
```

### Advanced Configuration

```bash
# With custom error handling and logging
python classify.py \
    --model "anthropic/claude-3-sonnet" \
    --n 3 \
    --on_fail reask \
    --retries 5 \
    --timeout 120 \
    --log-file "detailed.log" \
    --verbose

# Batch processing with custom policies
python classify.py \
    --model "meta/llama-2-70b-chat" \
    --n 1 \
    --on_fail random \
    --seed 42 \
    --sleep 1.0
```

## 🎛️ Command Reference

### Required Arguments
- `--model`: Replicate model ID (e.g., `owner/model[:version]`)

### Core Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n` | int | 1 | Number of independent runs |
| `--csv` | string | `../data/labeled_data.csv` | Path to input CSV |
| `--verbose` | flag | False | Enable detailed logging |

### Performance Tuning
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sleep` | float | 0.3 | Seconds between API calls |
| `--retries` | int | 3 | Retry attempts per article |
| `--timeout` | float | 60.0 | Maximum backoff time (seconds) |

### Error Handling
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--on_fail` | choice | "na" | Policy for unparseable outputs |
| `--seed` | int | 0 | Random seed for reproducibility |
| `--error_log` | string | auto | Custom error log path |
| `--log-file` | string | None | Detailed log file path |

### Failure Policies (`--on_fail`)
- **`na`**: Store as NA and log (recommended)
- **`d`**: Default to neutral (0.0)
- **`random`**: Assign random bias score
- **`reask`**: Attempt simplified re-prompt

## ⚖️ Bias Scoring System

### Analytical Approach
The system uses sophisticated prompt engineering to guide LLMs through a structured analysis:

1. **Left-Leaning Indicators**: Progressive language, social justice framing, climate urgency, wealth inequality focus, government regulation support
2. **Right-Leaning Indicators**: Conservative values, traditional themes, free market language, individual responsibility, limited government preference
3. **Tone Assessment**: Emotional vs factual language, loaded adjectives, source selection bias
4. **Precise Scoring**: Decimal precision to capture subtle nuances

### Scoring Scale

| Score Range | Classification | Description |
|-------------|----------------|-------------|
| -3.0 | Far Left | Revolutionary/extreme progressive |
| -2.0 | Strong Left | Clearly progressive agenda |
| -1.0 | Moderate Left | Noticeable liberal lean |
| -0.5 | Slight Left | Subtle progressive indicators |
| 0.0 | Perfectly Neutral | Rare - only if truly balanced |
| +0.5 | Slight Right | Subtle conservative indicators |
| +1.0 | Moderate Right | Noticeable conservative lean |
| +2.0 | Strong Right | Clearly conservative agenda |
| +3.0 | Far Right | Extreme conservative/nationalist |

### Anti-Zero-Bias Strategy
The system actively discourages lazy 0.0 ratings by:
- Explicit instructions to find subtle bias indicators
- Multiple prompts emphasizing nuanced analysis
- Examples demonstrating decimal precision
- Warning against "perfectly neutral" assumptions

## 📁 Output Structure

### Directory Organization
```
data/
├── labeled_data.csv              # Original dataset
├── predictions/
│   └── predictions_modelname.csv # Results with prediction columns
├── logs/
│   └── modelname_log.jsonl      # Detailed interaction logs
└── errors_modelname_timestamp.jsonl # Failed/unparseable responses
```

### Output Files

#### 1. Predictions CSV (`predictions/predictions_modelname.csv`)
Contains all original columns plus:
- **Single run**: `pred_modelname`
- **Multiple runs**: `pred_[1]_modelname`, `pred_[2]_modelname`, etc.

#### 2. Main Log (`logs/modelname_log.jsonl`)
Structured logs with:
```json
{
  "timestamp": "2024-09-05T10:30:15",
  "model": "meta/llama-2-70b-chat",
  "article_id": "row_42",
  "status": "success",
  "input_tokens": 1250,
  "output_tokens": 3,
  "parsed_code": "1.7",
  "variant_used": {"temperature": 0.8}
}
```

#### 3. Error Log (`errors_modelname_timestamp.jsonl`)
Failed attempts with:
```json
{
  "row_index": "42",
  "story_id": "story_12345",
  "model": "meta/llama-2-70b-chat",
  "raw_first": "I cannot provide bias ratings...",
  "news_title": "Article Title...",
  "timestamp": "2024-09-05T10:30:15"
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
REPLICATE_API_TOKEN=your-token-here

# Alternative (legacy)
REPLICATE_KEY=your-token-here
```

### Model Recommendations

#### Tier 1: High Performance
- `anthropic/claude-3-sonnet`: Excellent analytical depth
- `meta/llama-2-70b-chat`: Strong reasoning capabilities
- `mistralai/mixtral-8x7b-instruct-v0.1`: Fast and reliable

#### Tier 2: Balanced
- `meta/llama-2-13b-chat`: Good performance/cost ratio
- `microsoft/wizardlm-70b-v1.0`: Consistent outputs

#### Tier 3: Experimental
- Various fine-tuned models for specific bias detection

### Performance Optimization

#### For Large Datasets (1000+ articles)
```bash
--sleep 0.1 --retries 2 --timeout 30
```

#### For High Accuracy (Research)
```bash
--n 5 --on_fail reask --retries 5
```

#### For Cost Optimization
```bash
--n 1 --sleep 0.5 --on_fail na
```

## 🔍 Troubleshooting

### Common Issues

#### 1. API Key Problems
```bash
# Verify API key
echo $REPLICATE_API_TOKEN
# Or check .env file
cat .env
```

#### 2. Model Access Issues
```bash
# Test model access
python -c "import replicate; print(replicate.models.get('meta/llama-2-70b-chat'))"
```

#### 3. High Failure Rates
- Increase `--sleep` parameter
- Try different model
- Check `--verbose` output for specific errors
- Review error logs in detail

#### 4. Memory Issues
- Process smaller batches
- Reduce concurrent operations
- Monitor system resources

### Performance Issues

#### Slow Processing
- Decrease `--sleep` (if not rate-limited)
- Use faster models
- Reduce `--retries` for failed attempts

#### High API Costs
- Monitor token usage in logs
- Use smaller, efficient models
- Implement cost tracking

### Error Analysis

#### Check Error Logs
```bash
# View recent errors
tail -n 20 data/errors_*.jsonl

# Count error types
grep -o '"status":"[^"]*"' data/logs/*.jsonl | sort | uniq -c
```

#### Debug Specific Issues
```bash
# Run with maximum verbosity
python classify.py --model "your-model" --verbose --log-file debug.log

# Check specific row
grep "row_42" data/logs/*.jsonl
```

## 🏆 Best Practices

### Data Preparation
1. **Clean Input Data**: Remove malformed entries, ensure UTF-8 encoding
2. **Validate Required Columns**: Verify all required fields are present
3. **Handle Duplicates**: Consider story_id duplicates across sources
4. **Sample Testing**: Test with small batches before full runs

### Model Selection
1. **Start with Tier 1 Models**: Use proven high-performance models
2. **Consider Cost vs Quality**: Balance accuracy needs with budget
3. **Test Multiple Models**: Compare outputs across different LLMs
4. **Version Pinning**: Use specific model versions for reproducibility

### Production Deployment
1. **Implement Monitoring**: Track success rates and costs
2. **Set Rate Limits**: Respect API rate limits with appropriate `--sleep`
3. **Error Recovery**: Use `--on_fail reask` for critical applications
4. **Backup Strategies**: Regular saves and checkpointing

### Quality Assurance
1. **Multiple Runs**: Use `--n 3` or higher for reliability analysis
2. **Cross-Validation**: Compare with known ground truth when available
3. **Manual Review**: Spot-check predictions, especially edge cases
4. **Statistical Analysis**: Analyze inter-run consistency

## 📚 API Reference

### Core Functions

#### `classify.py main()`
Main entry point handling argument parsing and orchestration.

#### `call_model(model_id, title, text, row_num, run, log_path, verbose)`
Executes model inference with caching and error handling.

**Parameters:**
- `model_id`: Replicate model identifier
- `title`: Article headline
- `text`: Article content
- `row_num`: Row index for tracking
- `run`: Current run number
- `log_path`: Path for logging
- `verbose`: Enable detailed output

#### `extract_code(raw_response)`
Parses model output to extract numerical bias scores.

**Returns:** String representation of bias score or None

#### `ProgressTracker(total, description)`
Provides real-time progress updates with ETA calculation.

### Configuration Classes

#### `setup_logging(verbose, log_file)`
Configures structured logging with appropriate formatters.

#### `build_variants(user_prompt)`
Generates multiple input format variants for model compatibility.

### Utility Functions

#### `sanitize_model_id(model_id)`
Converts model names to safe filesystem names.

#### `col_name_for_k(k, model_id, total_runs)`
Generates appropriate column names for predictions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Replicate API for LLM access
- pandas community for data manipulation tools
- tiktoken for accurate token counting
- Political science research community for bias classification methodologies

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sakhadib/BiasClass--Classify-News-Bias-Using-LLM/issues)
- **Documentation**: This README
- **Examples**: See `data/` directory for sample formats

---

**⚠️ Important**: This tool is designed for academic research and analysis. Bias scores reflect statistical patterns in language, not absolute truth. Always consider context and validate results for your specific use case.
