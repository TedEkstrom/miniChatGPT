<br>INTRODUCTION<br/>
MiniChatGPT does not use a perfect training algorithm but is intended more as a way for me to learn what such training can look like and what the result will be. (I use Copilot to build the script for training and running the model.) Therefore, I cannot claim that the model that is good, which is also noticeable when testing it. I have used training data that I found from https://github.com/voidful/awesome-chatgpt-dataset and which seems to be a good and legitimate training set.

Here is a good article explaining the math behind ChatGPT: https://medium.com/@joshuaanang783/the-math-and-logic-behind-chatgpt-this-paper-is-all-you-need-57b82a0527f9

<br>REQUIREMENTS<br/>
To train it, you first need to install python, torch, sys, seaborn and keep all the files in the same folder. I use Ubuntu in a container to run mine on WSL2.

apt install python
pip install torch, sys, seaborn 

<img width="463" height="29" alt="image" src="https://github.com/user-attachments/assets/e3436c98-41ec-411b-b2aa-352ae02e3620" />


<br>TRAINING<br/>
You run train.py and then it will look for the file english_training_data.txt and train the model. During the training, it will create checkpoints that it can resume from.

python train.py

<img width="322" height="180" alt="image" src="https://github.com/user-attachments/assets/e108774f-8a28-448f-a2d7-298322bdee18" />
<img width="328" height="239" alt="image" src="https://github.com/user-attachments/assets/44deefc9-67cd-4f96-b650-c3b38bf378e4" />
<img width="1066" height="298" alt="image" src="https://github.com/user-attachments/assets/083a275c-9d22-42bd-9130-24b425c3fe49" />

It will also create the file: training_loss.png, attention.png, final_model.pt and chars.json when it is finished and an output from the word 'Hello'.

<img width="779" height="44" alt="image" src="https://github.com/user-attachments/assets/be658f55-a07a-4ba8-86ef-1baca3ce90c2" />


<img width="1009" height="62" alt="image" src="https://github.com/user-attachments/assets/551f54f4-df6a-4f35-a5e9-52ef60cc05d1" />

<br>RUN<br/>
If you want to test run it, use the file run.py. It will load the trained model along with characters. Keep in mind that the same settings must be made in run.py if you have changed train.py.

python run.py "Hello" 

<img width="395" height="40" alt="image" src="https://github.com/user-attachments/assets/67917dc8-b9ea-4326-8687-3fa21f968463" />

<br>EXTRACT<br/>
Since the format that comes from the normal training data is in json format. So I have chosen to convert it to string before so it doesn't take so long for each training.
python extract.py

