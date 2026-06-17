<br>INTRODUCTION<br/>
MiniChatGPT does not use a perfect training algorithm but is intended more as a way for me to learn what such training can look like and what the result will be. (I use Copilot to build the script for training and running the model.) Therefore, I cannot claim that the model that is good, which is also noticeable when testing it. I have used training data that I found from https://github.com/voidful/awesome-chatgpt-dataset and which seems to be a good and legitimate training set.

Here is a good article explaining the math behind ChatGPT: https://medium.com/@joshuaanang783/the-math-and-logic-behind-chatgpt-this-paper-is-all-you-need-57b82a0527f9

<br>REQUIREMENTS<br/>
To train it, you first need to install python, torch, sys, seaborn and I ran everything on Linux inside a container. After that, all files should be in the same folder.

apt install python
pip install torch, sys, seaborn 

<img width="463" height="29" alt="image" src="https://github.com/user-attachments/assets/e3436c98-41ec-411b-b2aa-352ae02e3620" />


<br>TRAINING<br/>
You run train.py and then it will look for the file english_training_data.txt and train the model. During the run, it will create checkpoints that it can resume from.

python train.py

<img width="322" height="180" alt="image" src="https://github.com/user-attachments/assets/e108774f-8a28-448f-a2d7-298322bdee18" />
<img width="328" height="239" alt="image" src="https://github.com/user-attachments/assets/44deefc9-67cd-4f96-b650-c3b38bf378e4" />
<img width="1066" height="298" alt="image" src="https://github.com/user-attachments/assets/083a275c-9d22-42bd-9130-24b425c3fe49" />

It will also create the file: training_loss.png, attention.png, final_model.pt and chars.json when it is finished and an output from the word 'Hello'.

<br>RUN<br/>
If you want to test run it, use the file run.py and type python run.py "Hello how are you" which will test the words "Hello how are you" in this case. Replace the word or sentence with what you want to test.

python run.py "Hello" 


