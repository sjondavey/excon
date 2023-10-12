import pytest
from src.valid_index import ValidIndex
from src.tree_tools import TreeNode, Tree

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

class TestTree:
    index_checker = create_excon_index_checker()

    def test_add_to_tree(self):
        tree = Tree("Excon", self.index_checker)
        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        with pytest.raises(ValueError):
            tree.add_to_tree(invalid_reference, heading_text='')

        #Check all nodes get added
        valid_index = 'G.1(C)(xviii)(c)(dd)(9)'
        tree.add_to_tree(valid_index, heading_text='Some really deep heading here')
        number_of_nodes = sum(1 for _ in tree.root.descendants) # excludes the root node
        assert number_of_nodes == 6

        #check that if a duplicate is added, it does not increase the node count
        sub_index = 'G.1(C)(xviii)'
        tree.add_to_tree(valid_index, heading_text='Some less deep heading here')
        number_of_nodes = sum(1 for _ in tree.root.descendants) # excludes the root node
        assert number_of_nodes == 6



    def test_get_node(self):
        tree = Tree("Excon", self.index_checker)
        invalid_reference = 'G.1(C)(xviii)(c)(DD)(9)'
        with pytest.raises(ValueError):
            tree.get_node(invalid_reference)
        invalid_reference = ''
        with pytest.raises(ValueError):
            tree.get_node(invalid_reference)
        
        assert tree.get_node("Excon") == tree.root
        assert tree.get_node("Excon").full_node_name == ""
        assert tree.get_node("Excon").heading_text == ""

        excon_description = "Exchange control manual hierarchy"
        tree.add_to_tree("Excon", heading_text=excon_description)
        assert tree.get_node("Excon").heading_text == excon_description

        valid_index = 'G.1(C)(xviii)(c)(dd)(9)'
        tree.add_to_tree(valid_index, heading_text='Some really deep heading here')
        assert tree.get_node(valid_index).heading_text == 'Some really deep heading here'
        sub_index = 'G.1(C)(xviii)'
        assert tree.get_node(sub_index).heading_text == ''
        tree.add_to_tree(sub_index, heading_text='Some less deep heading here')
        assert tree.get_node(sub_index).heading_text == 'Some less deep heading here'



