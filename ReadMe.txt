*****************************************************************************************************

                      SWAP ASSISTANT

*****************************************************************************************************

We created a speech based voice assistant named “SWAP” which can open the following applications 
through the voice commands.

Applications listed:
1.	Browser
2.	Face book
3.	Photos
4.	Paint
5.	Excel
6.	Word
7.	Notepad
8.	Music

========================================
PRE RECORDINGS
========================================

* 20 recordings were used for each word for training.

* 10 recordings were used for testing.

* folder name: my_recordings

* naming convention of each recording: word_utteranceNumber

=========================================
POINTS TO BE CONSIDERED WHILE EXECUTING
=========================================

* Change the global variable training to 1 to train all the words. ( By default, training part is skipped )

* As the models were already generated and training can take a lot of time, this part can be skipped and can proceed with the testing part.

* Universe and Codebook were generated using own recordings, so the accuracy for other recordings can be low.

* For a better accuracy for other voice data, codeook has to be provided and models to be created for each word again.

=======================================
Instructions to run the file
=======================================

* open the [Swap_Assistant.sln] file in Visual Studio 10+.

* Run the Swap_Assistant.cpp ( F5 ).

* Make sure all the required folders were copied correctly.

========================================
ACCURACY
========================================

Overall accuracy on pre recordings: 88%