# turkishSpeakerChatBot

General Definition:

This project is a Turkish speaker chatbot project which has three part voice recognation, nlp engine , text to speach

-> Voice Recognation Part

    * First we request an integer letter from user for determining the listening time of algorithm
    * Then we capture the voice with Google voice recognation service for python
   
   -> NLP Engine Part
   
    * In this section we are using a deeplearning technique and several other methods. 
       # For deep learning we used Keras framework
       # For steaming Word we used bag of words tecnique and snowball api
       # For neoral network we used Keras which is an open-source neural-network library written in Python.
         -  It is a high-level neural networks API developed with a focus on enabling fast experimentation.
         -  In our implementation we used sequential Keras model.
         -  For activation we used two different method which are Rectified Linear Unit(Relu) and softmax.
       # To start with our sequential model we add 4 layers.
       # Three of the four layers are an fully connected dense layer. Thus all of the perceptrons has connection to 
         each other and each one has different weight in order to distinguish the behavior of algorithm.
       # In the first layer we set parameters as follows, input shape from our stemmed(after data process, based on our data set we have 57 different words.) 
         trained data size. For activation we used ReLU algorithm.
       # After that we apply an drop out layer. The drop out layer for avoiding over-fitting situation. Based on our data set, 
         we consider 30 percent drop out is sufficient. In our data set we have 8 different class(tag). Because of that we add one softmax layer in terms of each          class.
       # In order to compute the quantity that a model should seek to minimize during training. We decide the loss function as ”categorical cross-entropy”.
       # In our model we set learning rate 0.001. After that we compile the model based on this parameters. To train our data set based on this parameters we use fit method.
         We set our data in to 8 piece of batch size with 200 epochs. The following code adjusted based on these explanations.
       # In our program we are training our data set at once. In each sequence we use bag of words .
       # in our current data set we have 57 stemmed different word. In bag of word method , we tokenize the given word after that we used snowball stammer for stemming the words.
         After that we enumerate the word in our trained data set array.
       # If we receive a match we set those word 1 if not we let 0. If all the words become zero algorithm returns a common response which is for directing user common path with suitable words.
         If we receive match/matches in array the algorithm try to predict values with Keras model.
       # For getting accurate results we set an probability threshold rate which has 85.
   
   -> Text To Speach Part
   
       # This part the last part of our program. Before entering the next cycle(waiting command from user) program extract sounds from NLP engines response.
       
How To Use Program:

  This program is written in Linux/Ubuntu Environment for windows please install a common python installer program (for exp Anaconda).
  
  For IDE we used PyCharm 
  
  To run program ,
  
      $ sudo apt update
      $ sudo apt install python3-pip
  
  Then install dependend libraries(You may use other tool rather than pip like conda etc.)
  
      $ pip3 install nltk OR $ conda install -c anaconda nltk
      $ pip3 install snowballstemmer OR $ conda install -c anaconda snowballstemmer
      $ pip3 install Keras OR $ conda install -c anaconda keras
      $ pip3 install SpeechRecognition
      $ pip3 install google-cloud-speech
  
