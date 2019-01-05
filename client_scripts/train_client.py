import client_utility as util
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import sys
import warnings
warnings.filterwarnings("ignore")

num_epoch = sys.argv[1]

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

def get_google_credentials(machine):
        """ Gather the google API credentials to access google sheets. """
        return ServiceAccountCredentials.from_json_keyfile_name('/home/pi/Desktop/fl_{}/gsheet_credentials.json'.format(machine), scope)

def save_results_to_google_sheets(machine, results):
    """ Save the results directly in Google sheets by interacting via the API using the gspread package.
        Please rename the relevant sheets and files based on personal usage. I was using sheet name as pi_testrun_results
        and one individual sheet for each client with the name as client_(client_number)"""
    credentials = get_google_credentials(machine)
    gc = gspread.authorize(credentials)
    file = gc.open("pi_testrun_results")
    wks = file.worksheet('client_{}'.format(machine))
    acc_list = ['acc']
    acc_list.extend(results['acc'])
    acc_list.append('{}_epoch_cycle_accuracy'.format(int(len(results['acc']))))
    acc_list.append(int((time.time())))

    loss_list=['loss']
    loss_list.extend(results['loss'])
    loss_list.append('{}_epoch_cycle_loss'.format(int(len(results['loss']))))
    loss_list.append(int(time.time()))

    wks.append_row(acc_list)
    wks.append_row(loss_list)

if __name__ == '__main__':
    # Each client is identified by a different number. for each client the identification number has to be changed
    # to 2,3,4 etc. based on the total number of clients.
    # TO DO: Replace the clients numbers by identification via the IP addresses.
    client_number = 1

    loaded_model =  util.load_master_model(client_number)
    gen_model,gen_results = util.train_model(loaded_model,client_number)  #enter client machine number
    model_path = util.get_model_path(client_number)
    model_name = util.get_model_name(client_number)
    gen_model.save(model_path + model_name)
    model_file = util.get_model_file(client_number)
    model_name = util.get_model_name(client_number)
    util.send_model_to_server(model_file,model_name)
    util.delete_local_model(model_file)
    save_results_to_google_sheets(client_number,gen_results)
