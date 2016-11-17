
import datetime
import pandas as pd
import time
def create_submission(test_preds,CONFIG):

    now = datetime.datetime.now()

    test_files = [im for im in os.listdir('../../data/test_stg1/')]
    submission = pd.DataFrame(test_preds, columns=CONFIG['DATA']['FISH_CLASSES'])
    submission.insert(0, 'image', test_files)
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    submission.to_csv('submissions/'+sub_file, index=False)