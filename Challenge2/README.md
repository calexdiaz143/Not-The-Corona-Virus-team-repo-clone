<center>
  <h1>Welcome to Hack the Machine Track 2</h1>
  <h2>Challenge 2: Dangerous Situtations</h3>
</center>

<h3> Introduction </h3>

In this challenge, you are expected to replace, or add to the set of general rules provided below:

<b>General rules:</b>

*	Wear a mask
*	Stay 6 feet from others
*	Avoid crowds
*	Avoid poorly ventilated spaces
*	Wash your hands often
*	Cover coughs and sneezes
*	Clean and disinfect frequently touched surfaces daily
*	Monitor your health daily
*	Get vaccinated

Your task for this challenge is to search the CDC's recommendations and guidelines for COVID-19 to find rules on how to behave during each of the prompted scenarios.

You will then be graded for how well you merge the set of rules for that individual scenario with general rules for COVID-19. You may add or replace rules from the general guidelines with rules for the specific scenario you are focused on in that section. All specific scenario guidelines will take precedence when conflicting with general guidelines. Your ability to find the supplemental rules and create a union set of the two sets of rules (the general rules and the rules you find from the CDC data) will be the basis upon which you are graded.

The CDC's data has two kind of markings to help you navigate the vast text file:
A *** marks the title of that page of information in the cdc guidance. Everything below that triple asterisk is directly related to the title
An example of this is "***Things to know about the COVID-19 pandemic" is followed by all information listed by the CDC as being important to know about the COVID-19 pandemic

The next example is "***isolate if you are sick" which is followed by information about why to isolate if you are sick in the pandemic.

We have included these markers to help you navigate the large dataset.

Another important marker in the dataset is the @ sign. This is included in the dataset to show important information that may be relevant to the challenge. The @ sign can point out important headers or marks when a list of important information is about to begin. Many important rules that could be relevant to part 1 of this challenge will be found near @ signs.

All rules that you submit to replace or modify the general rules must be actionable INSTRUCTIONS. For example: all of the general rules listed above are actions that an individual can execute in the real world. Things like: "wash your hands" and "Stay 6 feet away from others" can be carried out by a person in the real world.

If you choose to submit information as a rule that is not actionable it will NOT COUNT. Examples of submissions for rules that have no action:

* "Treatments Your Healthcare Provider Might Recommend if You Are Sick"
* "If Hosting"
* "If you must visit in-person, protect yourself and others"


The CDC's COVID-19 guidelines will be the source for developing supplemental rules that the contestant will use to modify the CDC's standing [rules for COVID-19](https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/prevention.html).


<h3> Challenge Instructions </h3>

1.	Sort through the CDC's recommendations/general guidance to find the rules that apply to your specific scenario.
2.	Create a union set of the general rules and the rules you found for #1 above. The rules you find in the CDC data will overlap or replace the general rules where possible. You will be assigned points based on your ability to replace and modify the general rules with the rules you have parsed from the CDC data.


<h3> Grading Criteria </h3>

*	A maximum of 18 rules may be added for any scenario. BUT that does not mean that each scenario requires 18 rules to be added.
*	There is a correct merger of the CDC guidelines and general rules for each scenario that has been created by the HACKtheMACHINE organizers. Your submission will be graded on its ability to match this ground truth merged set.
*	Adding as many rules as possible for each challenge will not yield more points because each situation has specific instructions that are best suited for that adverse scenario.


<h3> Example Situation </h3>

__"If you are traveling"__

<b>General rules:</b>

*	Wear a mask
*	Stay 6 feet from others
*	Avoid crowds
*	Avoid poorly ventilated spaces
*	Wash your hands often
*	Cover coughs and sneezes
*	Clean and disinfect frequently touched surfaces daily
*	Monitor your health daily
*	Get vaccinated

<b> 1. Rules parsed from the CDC's guidelines for COVID-19 for "If you are traveling". </b>

If you are eligible, get fully vaccinated for COVID-19. Wait 2 weeks after getting your second vaccine dose to travel-it takes time for your body to build protection after any vaccination.
Get tested with a viral test 1-3 days before you travel. Keep a copy of your test results with you during travel in case you are asked for them. Do NOT travel if you test positive.
Check travel restrictions before you go.
Wear a mask over your nose and mouth when in public settings. Masks are required on planes, buses, trains, and other forms of public transportation traveling into, within, or out of the United States and in U.S. transportation hubs such as airports and stations.
Avoid crowds and stay at least 6 feet/2 meters (about 2 arm lengths) from anyone who did not travel with you. It's important to do this everywhere - both indoors and outdoors.
Wash your hands often or use hand sanitizer (with at least 60% alcohol).
Bring extra supplies, such as masks and hand sanitizer.
Avoid contact with anyone who is sick.
Avoid touching your eyes, nose, and mouth.

<b> 2. Merged set of general and CDC rules: </b>

Wear a mask over your nose and mouth when in public settings. Masks are required on planes, buses, trains, and other forms of public transportation traveling into, within, or out of the United States and in U.S. transportation hubs such as airports and stations.
Avoid crowds and stay at least 6 feet/2 meters (about 2 arm lengths) from anyone who did not travel with you. It's important to do this everywhere - both indoors and outdoors.
Avoid contact with anyone who is sick.
Avoid poorly ventilated spaces
Wash your hands often or use hand sanitizer (with at least 60% alcohol).
Cover coughs and sneezes
Clean and disinfect frequently touched surfaces daily
Monitor your health daily
If you are eligible, get fully vaccinated for COVID-19. Wait 2 weeks after getting your second vaccine dose to travel-it takes time for your body to build protection after any vaccination.
Get tested with a viral test 1-3 days before you travel. Keep a copy of your test results with you during travel in case you are asked for them. Do NOT travel if you test positive.
Check travel restrictions before you go.
Bring extra supplies, such as masks and hand sanitizer.
Avoid touching your eyes, nose, and mouth.


<h3> Submission Instructions </h3>

For each rule that you will include in your final rule set you will label the left column with the situation you are applying the rule to and place the rule for that situation in the right column. You may place a maximum of 20 rules in your final rule set __FOR EACH SITUATION.__

<b>THE NAME OF YOUR CSV MUST BE "Challenge2_submission.csv".</b> This is what the automated submission validation script in GitHub Actions is looking for and also what we will look for in your repo at grading time.

| situation | rules |
|:---------:|:----------:|
| sick	| Rule goes here |
| sick	| Rule goes here |
| older_adult	| Rule goes here |
| older_adult	| Rule goes here |
| asthma	| Rule goes here |
| asthma	| Rule goes here |
| covid_with_newborn	| Rule goes here |
| covid_with_newborn	| Rule goes here |

Drop your CSV file into the following directory before 5 pm EST on Thursday the 25th under the /Challenge2/submission folder
  <h5>
    <b>
      Submission description requirement:
    </b>
  </h5>

You must also submit your team's submission description by 5pm EST on Thursday the 25th. See Submission_submission.docx which lays out the requirements to get full points for explanation.
Submit this file in the YOURTEAM/Challenge1/submission Directory

This is your team's required submission description document that must be turned in for each challenge.


<br>
  <p align="center">
    <img src="https://github.com/TRACK2-DATA-SCIENCE-2021/template-repo/blob/main/.github/images/Challenge2_image1.png" />
  </p>
</br>
