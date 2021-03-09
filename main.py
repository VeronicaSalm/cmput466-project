# --------------------------------------------------
# main.py
#
# Main entry point for the whole project.
# --------------------------------------------------

from settings import *
from DataManager import DataManager

def main():
    '''
    Driver code for the project.
    '''

    # Just simply initialize the data manager class for the
    # newsgroup dataset and load in the data.
    dm = DataManager(NEWSGROUP_DIR, 'newsgroup')
    dm.load_data()

    # See class definition for all methods, however you can do stuff like:
    print(dm.get_label(0))
    # To get a particular label, etc.


# Entry point to the program.
if __name__ == '__main__':
    main()

