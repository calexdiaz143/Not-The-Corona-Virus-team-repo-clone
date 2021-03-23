import pandas as pd
import argparse


def validate_csv_info(challenge_number, filename):
    '''
    Validation function to ensure that the submission is a csv file and contains the correct file name for the submission
    :param challenge_number: (int) Challenge number being submitted (1-3)
    :param filename: (str) The name of the file that was uploaded
    :return:
    '''
    file, ext = filename.split(".")
    if ext!="csv":
        return False, f"Incorrect file extension: {ext}, should be a .csv file"
    if file!=f"Challenge{challenge_number}_submission":
        return False, f"Incorrect file name: {file}, should be named 'Challenge{challenge_number}_submission.csv'"
    return True, ""

def validate_challenge1_submission(df):
    results_col = 'danger_level'
    all_expected_cols = ["scenario_id","danger_level"]
    if not all(column in all_expected_cols for column in df.columns):
        return False, f"Ensure that the columns in the submission csv contain the expected columns names: {all_expected_cols}," \
                      f"the submission file uploaded contained the following columns:  {list(df.columns)}"
    if any(df[results_col].isna()):
        return False, f"The submission contained {sum(df[results_col].isna())} missing/NA values, resubmit without missing submissions"
    if df[results_col].dtype!=int:
        return False, f"The datatype for the '{results_col}' column is {str(df[results_col].dtype)}, ensure that the submission" \
                      f"has been rounded and is integer form prior to submission"
    if not all(df[results_col].isin([1,2,3,4,5,6])):
        return False, f"The {results_col} should contain values between 1-6, but the submission contained values of {set(df[results_col])}"
    return True, ""



def parse_args():
    """
    Utility for args parsing
    @return: args object to pull command line arguments from (see defaults which are used if no value is passed in)
    """
    parser = argparse.ArgumentParser(description='Run LLM Inference')
    parser.add_argument('-s', '--submission-file',
                        default="",
                        help='relative path to the submission file Challenge1_submission.csv')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        submission_df = pd.read_csv(args.submission_file)
    except Exception as e:
        print("Unable to find the submission file")
        raise e
    success, message = validate_challenge1_submission(submission_df)
    if success:
        print(
            f"Successful in validating Challenge1 submission")
    else:
        raise Exception(f"Unsuccessful in validating Challenge1 submission with message: {message}")

