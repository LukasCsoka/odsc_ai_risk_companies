# AI Risk to Companies

Artificial intelligence (AI), as most new technologies during human history, is a double-edged sword. It is well-known, that AI can pose risks to society or individuals if it was not designed with care and checks. In this workshop, we will focus on AI risk to companies. Only a few business leaders have had the opportunity to sharpen their intuition and adapt their organization governance framework to track and mitigate these risks, starting from in-house AI-development efforts and finishing with evaluating and monitoring third-party AI tools.

By completing this workshop, you will develop an understanding of various vectors of AI risks to companies and be able to improve the governance inside your organization. Additionally, you will get hands-on experience on the problem of biased data and simulate an adversarial attack on a machine learning model.

### Lesson 1: AI Risk to Companies
In the beginning, we will familiarize ourselves with AI risks to companies, exploring different types of risks, such as lack of trust, bias in data, automating unethical actions, lack of monitoring and how how to setup the governance framework within your organization.

### Lesson 2: Bias in Data
AI uses large volumes of data and then, using algorithms, learns on training data from the patterns and takes actions on previously unseen data. We will explore hands-on, how AI produces faulty results when the data is biased or problematic.

### Lesson 3: Adversarial Attack
In the last lesson, we will simulate adversarial attacks, where we as the attacker will find a set of subtle changes to an input that will cause the target model to misclassify and provide the output we want.

### Requirements for Participants 
Lesson 2 and 3 will consists of going through interactive Jupyter notebook in Python language. You do not need to prepare anything, if you want to only watch, but I strongly suggest exploring these notebooks alone and playing with the code after the session to fully grasp these techniques.
Both sessions have separate requirements on installed libraries, therefore I suggest creating two virtual Python environments. Here are the options you have – all are equal:

- Use Conda, tutorial here: [Getting started with conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- Use Venv, tutorial here: [Installing packages using pip and virtual environments — Python Packaging User Guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
- Using other online services, here I list some free to use services (in no specific order):
-- Google Collaboratory (Colab)
-- Microsoft Azure Notebooks
-- Databricks
-- Kaggle Kernels
-- Binder
-- Datalore
--  CoCalc

I have used Datalore to create materials for this session, but these materials are completely system-agnostic, therefore feel free to use any of above mentioned options. I have used Datalore, because I wanted to test this service, but I usually work either using Databricks or combining Conda and virtual machines.

The GitHub repository with all materials contains three folders:
- AI Risk to Companies: presentation
- Bias in Data: materials for lesson 2
-Adversarial Attack: materials for lesson 3

The repository is subject to change. All the materials will be made available in final version just before the session. These materials will stay there for foreseeable future.
Folders for coding parts (lesson 2 and 3) contain also requirements.txt file. This requirements files list all libraries I had installed during the creation of these materials. As I used Datalore, which creates environment with lot of pre-installed libraries, you will not require all these libraries, but if you installed them, the materials will work for you without any issue. You can install these libraries using pip, either directly or by using command: `pip install -r /path/to/requirements.txt`
Both these lessons use also dataset, that will be downloaded in the first code cells inside respective Jupyter notebooks. If you do not like Jupyter notebook, I also provide .py version of code.

I wish you all the best during the conference and your lives! 
Stay safe during these hard times.
Lukas

