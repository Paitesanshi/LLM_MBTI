
## 1. Install

First, install all packages with:

```sh
pip install -r requirements.txt
```


## 2. Get MBTI for Single Character

run `get_llms_mbti.py` to download models and test target character's mbti. 

You can specify models by following code:

```python
if __name__ == '__main__':
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        LlamaTokenizer
    )

    models = [
        'baichuan-inc/Baichuan-7B',
        'bigscience/bloom-7b1',
    ]

    tokenizers = [
        'baichuan-inc/Baichuan-7B',
        'bigscience/bloom-7b1',
    ]
    character="Beethoven"

    ...
```

## 3. Get MBTI for 20 Famous Characters

run `eval_roles_mbti_zh.py` to test 20 famous characters' mbti with Chinese prompts.
run `eval_roles_mbti_en.py` to test 20 famous characters' mbti with English prompts.

The results are saved in `pop20_mbti_parse_zh.json` and `pop20_mbti_parse_en.json` respectively.
The overall results are summarized in Table `pop20_results_parse_mbti_zh.csv` and `pop20_results_parse_acc_zh.csv`.
The same applies to the English version.

## 4. Get MBTI for 16 MBTI Types

run `eval_general_mbti_zh.py` to test 16 MBTI types with Chinese prompts.
run `eval_general_mbti_en.py` to test 16 MBTI types with English prompts.

The results are saved in `16_mbti_zh.json` and `16_mbti_en.json` respectively.
The overall results are summarized in Table `16_results_mbti_zh.csv` and `16_results_mbti_acc_zh.csv`.
The same applies to the English version.