import pytest
import pandas as pd
from src.valid_index import ValidIndex
from src.file_tools import add_full_reference

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

