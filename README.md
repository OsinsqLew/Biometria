# VoiceAnalyzer

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/OsinsqLew/Biometria.git
cd Biometria
```

2. Set up the virtual environment:
```
python -m venv venv
source env/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```
pip install huggingface_hub numpy opencv-python
pip install insightface --only-binary :all:
pip install --upgrade insightface
pip install onnxruntime

```