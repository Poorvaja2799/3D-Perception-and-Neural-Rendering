# Project 5: Classifying Point Clouds Using PointNet

## Getting Started
  - See Project 0 for detailed environment setup.
  - Ensure that you are using the environment `cv_proj5`, which you can install using the install script `conda/install.sh`. 
 (Recommended)
- Alternative way to set up the environment (Recommended):
  1. make sure you are in your Project5/conda folder, then enter those commands in the terminal.
  2. ```mamba env create -f environment.yml```
  3. ```conda activate cv_proj5```
  4. go back to your Project5 folder, ```pip install -e .```
<!-- - If both methods above don't work well, you can try to create the environment by using conda and pip install:
  1. make sure you are in the Project5/conda folder, then enter those commands in the terminal.
  2. ```conda create -n cv_proj5 python=3.11.0```
  3. ```conda activate cv_proj5```
  4. ```pip install -r requirements.txt```
  5. go back to your Project5 folder, ```pip install -e .``` -->

---

## PACE Support
- We've added [PACE ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) support for this course which gives you access to GPUs. Refer to docs/PACE_guide.pdf for setup instructions.

## Logistics
- Submit via [Gradescope](https://gradescope.com).
- Additional information can be found in `docs/project-5.pdf`.

## Important Notes
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## Rubric

- +10 pts: Part 1 - Data Loader
- +10 pts: Part 2 - Baseline
- +20 pts: Part 3 - Simplified PointNet
- +10 pts: Part 4 - Analysis
- +5 pts: Part 5 - PointNet with TNet (Extra Credit)
- +50 pts: Report
- +5 pts: Report (Extra Credit)
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `output/`: directory containing trained model weights
    - `setup.cfg`: setup file for environment, do not need to change this file
2. `<your_gt_username>_proj5.pdf` - your report

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
