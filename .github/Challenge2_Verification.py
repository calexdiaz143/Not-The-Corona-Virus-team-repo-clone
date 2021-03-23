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
    results_col = 'rules'
    all_expected_cols = ["situation","rules"]
    situation_vals = ["sick","older_adult","asthma","covid_with_newborn"]
    if not all(column in all_expected_cols for column in df.columns):
        return False, f"Ensure that the columns in the submission csv contain the expected columns names: {all_expected_cols}," \
                      f" the submission file uploaded contained the following columns:  {list(df.columns)}"
    if not all(val in situation_vals for val in df['situation']):
        return False, f"Ensure that the situation column contains entries for all of the situation types: {set(situation_vals)}," \
                      f" the submission file uploaded contained the following entries:  {set(df['situation'])}"

    if any(df[results_col].isna()):
        return False, f"The submission contained {sum(df[results_col].isna())} missing/NA values, resubmit without missing submissions"

    results_list = df[results_col].to_list()
    # Check to ensure that the dtype across the results column (rules) is a string, or 'O' (object)
    if df[results_col].dtype!='O':
        return False, f"The datatype for the '{results_col}' column is {str(df[results_col].dtype)}, ensure that the submission" \
                      f" has string values across the board for the column {results_col}  prior to submission"
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
            f"Successful in validating Challenge2 submission")
    else:
        raise Exception(f"Unsuccessful in validating Challenge2 submission with message: {message}")

