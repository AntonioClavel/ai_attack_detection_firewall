# AI Attack Detention Firewall

Based on the CIC-UNSW-NB15 dataset, this project is centered around building and deploying a hybrid machine learning model that can identify if an instance of network traffic is or not an attack, and based on the attack subdivisions that are provided in the dataset, guess which type we are facing.

To use/test it:
- 1: Clone this project into any machine.
- 2: The dataset is too large for Github, so go to "https://cicresearch.ca//CICDataset/CIC-UNSW/", fill out the information, and download "Data.csv" and "Label.csv", adding them to the Dataset folder of your cloned project.
- 3: Download Docker Desktop, and open it.
- 4: Start a new terminal inside Docker Desktop, and execute "cd (The directory you have cloned this project into)".
- 5: Execute "docker-compose up --build", wait until the terminal says that the project is up and ready to use.
- 6: Now the project is ready to go. Since we do not have live data connections, we will analyze our model by using rows of the dataset. Write "http://localhost:8501" on your prefered browser.
- 7: Select the amount of random rows you want to analyze, and hit the button below.
- 8: The analysis gets displayed for further evaluation, you can repeat step 6 as much as you want and recieve different rows every time.

This is an end-of-degree project for ETSISI-UPM.
