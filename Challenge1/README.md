<center>
  <h1>Welcome to Hack the Machine Track 2</h1>
  <h2>Challenge 1: Spectrum Labelling</h3>
</center>

<h3> Introduction </h3>

In this challenge, you will be assigning a level of risk to specific scenarios according to COVID-19 guidelines.
Each scenario is a short 1-3 sentence description of a possible situation that an individual is navigating.

Your job in this challenge: assign each scenario a level of risk from 1-6, with 1 being completely safe and 6 being extremely dangerous. Danger labelling scale:

Completely Safe (1) &rarr; Mostly Safe (2) &rarr; Moderately Safe (3) &rarr; Ill advised (4) &rarr; Dangerous (5) &rarr; Extremely Dangerous (6)

A training-test list of scenarios will be available to you so that you may tune and improve your algorithmsover the course of the event. You will be given a training data set, a test data set which does not contain the correct label.

<h3> Example Situation: </h3>

_"A 22-year-old college student would like to go to a beach party with their friends. The party will be in the open air with direct sunlight. Masks are not required. Drinks and food will be provided and served in a shared family meal context."_

The assessed label of this situation is a 3 on the spectrum of safe to dangerous.


<h3> Scoring </h3>

In order to receive points, __the methodology and algorithms__ behind your solutions must be provided for evaluation. This allows you to earn points even if you cannot create the physical solution you propose.
Up to 85 points will be awarded for getting the correct solution to the games along with a comprehensive write up that shows how you arrived at your solution. In order to receive points, the methodology, and algorithms behind your solutions must be provided for evaluation.
Up to 15 points will be awarded by a panel of judges for the creativity of the completed or proposed solution and for ideas about the application of AI ethics that might be necessary to use your solution in real-world scenario.


<h3> Submitting </h3>

To submit for this challenge you will commit a CSV file to the submission folder of the Challenge1 folder in your team's GitHub.

<b>THE NAME OF YOUR CSV MUST BE "Challenge1_submission.csv".</b> This is what the automated submission validation script in GitHub Actions is looking for and also what we will look for in your repo at grading time.

The CSV file will be submitted with the following format with your guesses for the test set of data that does not have labels (included in the Challenge's "data" folder).

| scenario_id | danger_level |
|:-----------:|:------------:|
| 1 | 2 |
| 2 | 5 |
| 3 | 6 |
| 4 | 3 |
| 5 | 1 |
| ... | ... |
| ... | ... |
| ... | ... |


<h4>
  <b>
    Example of a correct submission for challenge 1:
  </b>
</h4>



<br>
  <p align="center">
    <img src="../.github/images/Challenge1_image1.png" />
  </p>
</br>


<h4>
  <b>
    Submission description requirement:
  </b>
</h4>

You must also submit your team's submission description by 5pm EST on Thursday the 25th. See Submission_submission.docx which lays out the requirements to get full points for explanation.

Submit this file in the YOURTEAM/Challenge1/submission Directory
This is your team's required submission description document that must be turned in for each challenge.

<br>
  <p align="center">
    <img src="../.github/images/Challenge1_image2.png" />
  </p>
</br>
