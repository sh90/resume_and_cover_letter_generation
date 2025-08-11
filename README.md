# Personalized Resume & Cover Letter Generator

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
5. `streamlit run app/app.py`
6. We will also perform fine-tuning of model today. Set `GEN_MODEL=ft:gpt-4o-mini-2024-07-18:personal:resume-cover-ft:C3HhrPnR` post finetuning job in .env
6. Post finetuning is done, we will run `streamlit run scripts/ab_test_UI.py`
   
## Features:
1. Upload/Paste Job description and Resume Deatils
2. Provide Few-shot Examples ( Optional )
3. Generate tailored bullet pointers for resume
4. Generate tailored cover letter
5. Enhace the ouptput with few-shot examples
6. Fine-tuning of GPT-4o-min
7. Compare performance and store the results
8. Key metrics Definition:
   
   a. keyword_coverage
  Fraction of important JD terms that show up in the model’s output (0–1 scale). We extract likely keywords (TitleCase, long words, common skills) and check presence. Higher = better JD alignment.
  
   b. quantify_score
  Measures how “quantified” the writing is by counting numbers, %, and impact verbs per line. Higher means more measurable outcomes (great for resume bullets).
  
   c. length_ok (cover letters only)
  Boolean check that the letter lands in a sensible range (default 120–200 words). Helps keep letters concise yet substantial.
  
   d. composite_score
  Single yardstick combining the above: for bullets → ~60% keyword coverage + 40% quantification; for cover letters → ~50% keyword + 40% quant + small bonus if length_ok. Capped at 1.0 for easy comparison.
   
## Data preparation for fine-tuning
Steps: 
1. python scripts/prep_dataset.py
This step will create finetune.jsonl
2. python scripts/run_finetune.py
This will execute a finetuning job and will create data/tuned_model.txt
