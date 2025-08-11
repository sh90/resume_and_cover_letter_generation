# Customer Service Agent
### IDE 
Download Pycharm: https://www.jetbrains.com/pycharm/download/?section=windows or https://www.jetbrains.com/pycharm/download/?section=mac 

### How to use Pycharm
1. Open pycharm and create new project
 <img width="248" height="310" alt="image" src="https://github.com/user-attachments/assets/bddc76a3-f0b5-4c47-b74d-2a6be0dff2c9" />

2. Create Virtual env with python3.12
<img width="637" height="320" alt="image" src="https://github.com/user-attachments/assets/a338e946-785d-4caa-9d2e-6beb5a93a6b9" />



### Setup
1. Open Pycharm and create virtual environment using python3.12
2. `pip install -r requirements.txt`
3. [Get OpenAI API key](https://platform.openai.com/)
4. Create .env file inside project and paste `OPENAI_API_KEY="sk-..."` in your .env
5. `streamlit run app.py`
6. 
## Data preparation for fine-tuning
Steps: 
1. python scripts/prep_dataset.py
2. wc -l data/finetune.jsonl
3. head -n 2 data/finetune.jsonl

This step will create finetune.jsonl
