import time
import subprocess
from subprocess import PIPE
from datetime import timedelta
from IPython.display import clear_output


COMP_NAME = "nbme-score-clinical-patient-notes"

CHECK_INTERVAL = 1  # minutes
MAX_HOUR = 9


def check_state(find_name=None):
    proc = subprocess.run(f"kaggle competitions submissions -v {COMP_NAME}",
                          shell=True, stdout=PIPE, stderr=PIPE, text=True)
    ret = proc.stdout

    keys = ret.split('\n')[0].split(',')
    if find_name:
        find_sub = [r.split(',') for r in ret.split('\n')[1:] \
                    if r.split(',')[0] == find_name][0]
        dic_ret = {k:v for k,v in zip(keys, find_sub)}
    else:
        latest_sub = ret.split('\n')[1].split(',')
        if len(latest_sub) > 1:
            dic_ret = {k:v for k,v in zip(keys, latest_sub)}
        else:
            dic_ret = {'fileName':None}
    return dic_ret

def check_submit_time():
    start_time = time.time()

    request_num = int(MAX_HOUR * 60 / CHECK_INTERVAL)+1
    find_name = None
    for _ in range(request_num):
        dic_ret = check_state(find_name)

        if find_name is None:
            find_name = dic_ret['fileName']

        run_time = time.time() - start_time
        td = timedelta(seconds=int(run_time))

        if dic_ret['fileName'] is None:
            print("You don't have submit. Do your best.")
            return
        elif dic_ret['status'] == 'complete':
            if dic_ret['publicScore'] == 'None':
                mess = f"Submit Finish!\n"\
                       f"  **Error may have occurred**\n"\
                       f"  Exp = {dic_ret['fileName']}\n"\
                       f"  State = {dic_ret['status']}\n"\
                       f"  PubScore = {dic_ret['publicScore']}\n"\
                       f"  Time = {td}"
            else:
                mess = f"Submit Finish!\n"\
                       f"  Exp = {dic_ret['fileName']}\n"\
                       f"  State = {dic_ret['status']}\n"\
                       f"  PubScore = {dic_ret['publicScore']}\n"\
                       f"  Time = {td}"
            # message_to_slack(mess)
            clear_output(wait=True)
            print(mess)
            return
        else:
            mess = f"Submitting...\n"\
                   f"  Exp = {dic_ret['fileName']}\n"\
                   f"  State = {dic_ret['status']}\n"\
                   f"  Running time = {td}"
            clear_output(wait=True)
            print(mess)

        time.sleep(CHECK_INTERVAL*60)

    mess = f"Submit Timeout...\n"\
           f"  Exp = {dic_ret['fileName']}\n"\
           f"  State = {dic_ret['status']}\n"\
           f"  Time = {td}"
    # message_to_slack(mess)
    clear_output(wait=True)
    print(mess)
    return

# format = hh:mm:ss
check_submit_time()