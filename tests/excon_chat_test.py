import src.excon_chat    # The code to test

def test_increment():
    assert src.excon_chat.increment(3) == 4

# This test is designed to fail for demonstration purposes.
def test_decrement():
    assert src.excon_chat.decrement(3) == 2