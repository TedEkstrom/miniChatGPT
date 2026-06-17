INTRODUCTION
MiniChatGPT does not use a perfect training algorithm but is intended more as a way for me to learn what such training can look like and what the result will be. Therefore, I cannot claim that the model that is picked up is good, which is also noticeable when testing it. I have used training data that I found from ... and which seems to be a good and legitimate training set.

REQUIREMENTS
To train it, you first need to install python, .... and I ran everything on Linux inside a container.

After that, all files should be in the same folder.

TRAIN
You run training.py and then it will look for the file english_training_data.txt and train the model. During the run, it will create checkpoints that it can resume from.

It will also create the file: training_loss.png, attention.png, final_model.pt and chars.json when it is finished and an output from the word 'Hello'.

RUN
If you want to test run it, use the file run.py and type python run.py "Hello how are you" which will test the words "Hello how are you" in this case. Replace the word or sentence with what you want to test.
