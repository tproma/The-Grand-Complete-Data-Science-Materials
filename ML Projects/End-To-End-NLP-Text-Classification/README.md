# End-To-End-NLP-Project-News-Article-Sorting

- [GitHub](https://github.com/tproma)
- [LinkedIn](https://www.linkedin.com/in/tanjinaproma/)


### About The Project:
The primary objective of this project is to develop an automated system capable of accurately classifying news articles into predefined categories using state-of-the-art NLP models. The project involves fine-tuning bert-base-uncased on the collected dataset to adapt its knowledge to the specific classification task. This step enables the model to learn the nuances and patterns within the news articles.


### Tech-stack used
- Python
- PyTorch
- Hugging Face Transformers Library
- Docker 
- Flask
- GitHub Actions
- AWS cloud Services


### Step 1: Create condsa environment
```
conda create -n textSort python=3.8 -y
```
```
conda activate textSort
```

### Step 2: Install the requirements
```
pip install -r requirements.txt
```

# For pytorch cuda version
``` 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```
pip install -U scikit-learn
```

#Necessary for running the training arguments
```
pip install --upgrade accelerate
pip install -y transformers accelerate
pip install transformers accelerate
```

### To run the flask app
```
python app.py
```


## Workflows
- update config.yaml
- update  params.yaml
- update entity
- update the configuration manager in src config
- update the components
- update the pipelines
- update main.py
- update app.py


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/text-sort

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

