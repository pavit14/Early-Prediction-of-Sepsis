import pytest
import ..src.load as load
from mock import patch

@patch('load.os.path.endswith')
@patch('load.pd.read_csv')
@patch('load.os.path.splitext')
@patch('load.os.path.join')
@patch('load.os.path.dirname')
@patch('load.pd.concat')

def test_load(mock_endswith, mock_read_csv, mock_splitext, mock_join, mock_dirname, mock_concat):
    # arrange
    # always return true for isfile
    data_folder = load.os.path.join(load.os.path.dirname(__file__), "../data/training_setA")
    with patch('os.listdir') as mocked_listdir:
        mocked_listdir.return_value = data_folder
    load.os.path.endwsith.return_value = '.psv'
    filename = 'file.psv'

    # act
    _ = load.load_data()

    dataframes=[]

    # assert
    # check that read_csv is called with the correct parameters
    data=load.pd.read_csv.assert_called_once_with(load.os.path.join(data_folder, filename), delimiter='|')
    data['patientid'] = load.os.path.splitext(filename)[0]
    combined_data = load.pd.concat(dataframes, ignore_index=True)