<center>
  <h1>Welcome to Hack the Machine Track 2</h1>
  <h2>Challenge 3: Resource Optimization</h3>
</center>

<h3> Introduction: </h3>

In this challenge, you will design an algorithm to schedule treatment for the crew of the USS GHOST who may become ill with COVID-19. Every day the ship tests all 300 Sailors in the crew for COVID-19. When a crew member tests positive for COVID-19 the ship's hospital corpsman assigns a rating to indicate how fast the Sailor's health is declining.

<h3> Information: </h3>

*	You are starting out with a cohort of Sailors who have contracted COVID-19 on day 1 but you are expecting a surge in cases throughout the challenge.
*	Each sailor has a health level and a rate of decline level assigned to them. __The essence of this challenge is to maximize the health score of as many of the Sailors as possible.__
*	Sailors are tested each day for COVID-19. Some Sailors are healthier than others and so will have a higher base health score.  Sailors initial health can be any number between 1 and 100.  When a Sailor tests positive for COVID-19 they are assigned a rate of health decline.
*	You have 30 beds at your disposal. Each sailor who is allocated a bed will receive a base boost of 5 health points. Patients must be allocated a bed to receive other treatments like a ventilator or supplemental oxygen.
*	The health score at the end of each day is calculated by summing health, bed bonus, rate of health decline, ventilator bonus, etc.
*	If a sailor's health is 0 or below at the time of submission, the sailor dies, and you receive a penalty of -100 points.

<h3> Resources: </h3>

*	If you over-allocate resources to a particular sailor and push their total health score over 100, it does not contribute to your overall score. A score of 105 and a score of 100 in the health category are both worth a max of 100 points.
*	Resources are either reusable or one-time use.  Reusable resources can be re-allocated each day of the challenge.  One-time use resources can be allocated only once for the duration of the challenge.  If 5 Dexamethasone are applied on day 1 only 15 Dexamethasone are available for days 2 and 3.
*	The resources that are available for your allocation include:
    * Reusable:
      * 10 Ventilators which add 30 health but cannot be combined with Supplemental oxygen masks
      * 10 Supplemental oxygen masks which add 20 health but cannot be combined with Ventilators
    * One-time use:
	    * 10 Plasma which adds 15 health
	    * 7 Remdesivir which adds 30 health
	    * 20 Dexamethasone which adds 25 health
	    * 10 Casirivimab which adds 15 health
	    * 17 Chloroquine which adds 10 health
*	Be warned: each day's decision on how to allocate one-time use resources affects the next day's availability. You do not receive more resources during the challenge, so over or under allocation can cause you problems later in the challenge.


<h3> Data Release Schedule: </h3>

For this challenge you will receive one data document from each day of the challenge that gives you information about new sick patients that have been admitted to your med bay. This data will include information about the sailor's ID, Health, and Rate of health decline for that day.

<h3> Bonus Points:</h3>

*	For each resource not allocated you will receive half that resource's health bonus as points towards your total score for that day.
    * For example: if you have 7 Remdesivir available which each add 30 health points and you only use 5 on your first day. Then you will receive 30(.5)*2 = 30 bonus points for conserving 2 Remdesivir treatments that day.  
*	_NOTE: There is no bonus for conserving beds._

<h3> Submission Instructions</h3>

You will submit 3 files to the Challenge 3 submission folder, one for each day of data that assigns patients to beds and allocates available resources.

*	You will be required to submit your answer for day 1 by 7pm EST of day 1. Day 2 and day 3 submissions must be made by 5pm EST on their respective nights.
*	You will not be able to receive points for a day if you do not submit by the deadline for that day.
    * For example: if you submit all 3 days worth of data on day 3 you will not receive any points for days 1 or 2.

THE NAME OF EACH DAY'S CSV MUST FOLLOW A SPECIFIC FORMAT:
* day 1 submission should be "Challenge3_Day1Submission.csv".
* day 2 submission should be "Challenge3_Day2Submission.csv".
* day 3 submission should be "Challenge3_Day3Submission.csv".

This naming is what the automated submission validation script in GitHub Actions is looking for and also what we will look for in your repo at grading time.

  <h4> Daily score keeping: </h4>

  Each day of the event the top scores on challenge 3 will be released on a leader board to be announced during our scheduled broadcasting. The top scoring team's ranking will be based on the total health of all their Sailor's up to that point in the challenge.

  <h4>  Submitting:</h4>

  Each day of the challenge you must submit that day's submission in CSV form by 5PM EST.

A sample submission is in the submissions folder which can be used as a template showing how to format your submission files correctly.  Column names in your team's submission files must be the same as those shown below and in the sample submission.  Note the capitalization of the first letter in each column.

Example submission:


| ID | Health | Rate of health decline | Bed | Ventilator | Remdesivir | Dexamethasone | Plasma | Casirivimab | Supplemental oxygen | Chloroquine | Total |
|:------------:|:------:|:----------------------:|:----:|:----------:|:------:|:----------:|:-------------:|:--------:|:------------:|:--------:|:-----:|
| 1 | 90 | -82 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 88 |
| 2 | 8  | -5  | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 73 |
| 3 | 70 | -55 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 95 |
| . |  . | .   | . | . | . | . | . | . | . | . | . |
| . |  . | .   | . | . | . | . | . | . | . | . | . |
| . |  . | .   | . | . | . | . | . | . | . | . | . |


Tuesday the 23rd's CSV submission must be titled:" Challenge3_Day1Submission.csv"

Wednesday the 24th's CSV submission must be titled:" Challenge3_Day2Submission.csv"

Thursday the 25th's CSV submission must be titled:" Challenge3_Day3Submission.csv"

You will submit each day's data to the /Challenge3/submissions folder in your team's GitHub repository.


  <h3>  Submission description requirement:</h3>

You must also submit your team's submission description by 5pm EST on Thursday the 25th. See Submission_submission.docx which lays out the requirements to get full points for explanation.
Submit this file in the YOURTEAM/Challenge1/submission Directory
This is your team's required submission description document that must be turned in for each challenge.

<br>
  <p align="center">
    <img src="https://github.com/TRACK2-DATA-SCIENCE-2021/template-repo/blob/main/.github/images/Challenge3_image1.png" />
  </p>
</br>
