import pytest
import pandas as pd
import os
import fnmatch

from src.valid_index import ValidIndex
from src.file_tools import add_full_reference, read_processed_regs_into_dataframe

def create_excon_index_checker():
    exclusion_list = ['Legal context', 'Introduction']
    excon_index_patterns = [
            r'^[A-Z]\.\d{0,2}',
            r'^\([A-Z]\)',
            r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii)\)',
            r'^\([a-z]\)',
            r'^\([a-z]{2}\)',
            r'^\((?:[1-9]|[1-9][0-9])\)',
        ]
    return ValidIndex(regex_list_of_indices=excon_index_patterns, exclusion_list=exclusion_list)


def test_add_full_reference():    
    index_checker = create_excon_index_checker()

    # Sample DataFrame
    df = pd.DataFrame({
    'Indent':    [ 0,    0,    1,     2,      3,     2,     2],
    'Reference': ['A.1', '', '(B)', '(xx)', '(c)', '(xxi)', '']
    })
    add_full_reference(df, index_checker)
    assert df.loc[0, 'full_reference'] == 'A.1'
    assert df.loc[1, 'full_reference'] == 'A.1'
    assert df.loc[2, 'full_reference'] == 'A.1(B)'
    assert df.loc[3, 'full_reference'] == 'A.1(B)(xx)'
    assert df.loc[4, 'full_reference'] == 'A.1(B)(xx)(c)'
    assert df.loc[5, 'full_reference'] == 'A.1(B)(xxi)'
    assert df.loc[6, 'full_reference'] == 'A.1(B)(xxi)'

    df_with_indent_reference_mismatch = pd.DataFrame({
    'Indent':    [ 0,    0,    1,     2,      3,     2,     2],
    'Reference': ['A.1', '', '(B)', '(xx)', '(c)', '(d)', ''],
    'Text':      ['1',   '2','3',   '4',   '5',   '6',   '7']
    })
    with pytest.raises(ValueError):
        add_full_reference(df_with_indent_reference_mismatch, index_checker)

def test_read_processed_regs_into_dataframe():
    index_checker = create_excon_index_checker()
    non_text_labels = ['Table', 'Formula', 'Example', 'Definition']
    dir_path = './inputs/'
    file_list = []
    for root, dir, files in os.walk(dir_path):
        for file in files:
            str = 'excon_manual*.txt'
            if fnmatch.fnmatch(file, str):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    df_excon, non_text = read_processed_regs_into_dataframe(file_list=file_list, valid_index_checker=index_checker, non_text_labels=non_text_labels)
    assert len(df_excon) == 2607
    assert len(non_text['Table']) == 100
    assert len(non_text['Definition']) == 4
