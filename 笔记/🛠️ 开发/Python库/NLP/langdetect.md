
# langdetect

langdetect Library in Python

langdetect is a popular Python library for detecting the language of text. It's a Python port of Google's language detection library, providing fast and accurate language identification capabilities.

Key Features
- Multi-language Support: Detects over 55 languages
- High Accuracy: Based on Google's language detection algorithm
- Fast Performance: Optimized for quick detection
- Simple API: Easy to use with minimal setup
- Probabilistic Results: Returns confidence scores for detections

## Installation
```shell
pip install langdetect
```

## Basic Usage
### Simple Language Detection
```python
from langdetect import detect

# Detect language
text = "Hello, how are you today?"
language = detect(text)
print(language)  # Output: 'en'
```

Detection with Confidence Scores
```python
from langdetect import detect_langs

# Get language with probability
text = "Bonjour, comment allez-vous?"
languages = detect_langs(text)
print(languages)  # Output: [fr:0.999999999999972]
```

```python
from langdetect import detect_langs

# Text with mixed languages
text = "I love programming. Me encanta programar."
languages = detect_langs(text)
for lang in languages:
    print(f"Language: {lang.lang}, Probability: {lang.prob}")
```


