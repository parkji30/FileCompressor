"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode

# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    >>> byte_to_bits(1)
    '00000001'
    >>> byte_to_bits(2)
    '00000010'

    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    return_dictionary = {}
    for b in text:
        if b in return_dictionary:
            return_dictionary[b] += 1
        else:
            return_dictionary[b] = 1
    return return_dictionary

# Used to undertand the Huffman Algorithm
#  https://en.wikipedia.org/wiki/Huffman_coding
#
# https://www.youtube.com/watch?v=dM6us854Jk0


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    >>> freq1 = {3:20, 2:11, 10:1}
    >>> result3 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(None, \
    HuffmanNode(2), HuffmanNode(10)))
    >>> result4 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(None, \
    HuffmanNode(10), HuffmanNode(2)))
    >>> result5 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(10), \
    HuffmanNode(2)) , HuffmanNode(3))
    >>> result6 = HuffmanNode(None, HuffmanNode(None, HuffmanNode(2), \
    HuffmanNode(10)) , HuffmanNode(3))
    >>> t1 = huffman_tree(freq1)
    >>> t1 == result3 or t1 == result4 or t1 == result5 or t1 == result6
    True
    >>> freq2 = {1:100}
    >>> t2 = huffman_tree(freq2)
    >>> t2 == HuffmanNode(1)
    True
    """
    # for this algorithm you we first make leaf nodes of all the symbols in the
    # freq_dict and stored these nodes in a tuple where the value in index
    # 1 of the tuple is the frequency of the node.
    # then while the size fo the list is greater than one we create new nodes
    # with the first two value in the list update the frequency value (value at
    # index 1 of the tuple) and append the tuple back into the list.
    # in the end we have a tuple of the root node at index 0 of the last tuple
    # so we return that.

    leaf_nodes = [(HuffmanNode(symbol, None, None), freq_dict[symbol],) for
                  symbol in freq_dict]
    leaf_nodes.sort(key=get_key_value)
    while len(leaf_nodes) > 1:
        left, right = leaf_nodes.pop(0), leaf_nodes.pop(0)
        branch = (HuffmanNode(None, left[0], right[0]), left[1] + right[1],)
        leaf_nodes.append(branch)
        leaf_nodes.sort(key=get_key_value)
    return leaf_nodes[0][0]


def get_key_value(obj):
    """Returns the object at index 1 of the Object y.

    @param Object obj: The object being indexed.
    @rtype: Object

    >>> l = [n for n in range(10)]
    >>> get_key_value(l)
    1
    """
    return obj[1]


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> freq1 = {3:20, 2:11, 10:1}
    >>> t1 = huffman_tree(freq1)
    >>> d2 = get_codes(t1)
    >>> d2 == {3:'1', 2:'01', 10:'00'}
    True
    """
    return helper_get_codes(tree)


def helper_get_codes(tree, prefix=''):
    """
    A helper for the function get_codes that allows us to keep track of the
    prefix. This function returns a dictionary with the keys being symbols of
    the tree and the values being the prefix code generated by navigating the
    tree.

    @param HuffmanNode tree: The root node of the tree
    @param str prefix: the prefix code that we have generated by traversing down
    the tree from the root node
    @rtype: dict{int:str}
    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = helper_get_codes(tree)
    >>> d == {2: '1', 3: '0'}
    True
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(None, \
    HuffmanNode(10), HuffmanNode(2)))
    >>> d1 = helper_get_codes(result1)
    >>> d1 == {3:'0', 10:'10', 2:'11'}
    True
    >>> t = HuffmanNode(3)
    >>> d2 = helper_get_codes(t)
    >>> d2 == {3:'0'}
    True
    >>> z = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
    HuffmanNode(100)), HuffmanNode(None, HuffmanNode(None, HuffmanNode(None, \
    HuffmanNode(8), HuffmanNode(11)), HuffmanNode(30)), HuffmanNode(10)))
    >>> d3 = helper_get_codes(z)
    >>> d3 == {3:'00', 100:'01', 8:'1000', 11:'1001', 30:'101', 10:'11'}
    True
    """
    result_dictionary = {}
    if tree.is_leaf():
        result_dictionary[tree.symbol] = '0'
    else:
        if tree.left.is_leaf():
            result_dictionary[tree.left.symbol] = prefix + '0'
        else:
            result_dictionary.update(helper_get_codes(tree.left, prefix + '0'))

        if tree.right.is_leaf():
            result_dictionary[tree.right.symbol] = prefix + '1'
        else:
            result_dictionary.update(helper_get_codes(tree.right, prefix + '1'))
    return result_dictionary


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> tree2 = HuffmanNode(None, HuffmanNode(100), HuffmanNode(200))
    >>> tree3 = HuffmanNode(None, tree2, tree)
    >>> number_nodes(tree3)
    >>> tree3.left.number
    0
    >>> tree3.right.number
    3
    >>> tree3.number
    4
    >>> tree3.right.right.number
    2
    >>> tree3.right.left.number
    1
    """
    nodes_in_post_order = create_post_list(tree)
    for index in range(len(nodes_in_post_order)):
        nodes_in_post_order[index].number = index


def create_post_list(tree):
    """ Recursiive helper function to label the internal nodes of tree. This
    function produces a list of the internal Huffman nodes as they appear in t
    he post order traversal of the tree.

    @type tree: Tree
    @rtype: list[HuffmanNode]
    >>> tree = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(2)), HuffmanNode(3))
    >>> create_post_list(tree)
    [HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None)),\
 HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None),\
 HuffmanNode(2, None, None)), HuffmanNode(3, None, None))]
    """
    result = []
    if tree is None:
        pass
    else:
        if tree.is_leaf():
            pass
        else:
            if not tree.left.is_leaf():
                result += create_post_list(tree.left)
            if not tree.right.is_leaf():
                result += create_post_list(tree.right)
            result += [tree]
    return result


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> get_codes(tree) == {9: '1', 2: '01', 3: '00'}
    True
    >>> avg_length(tree, freq)
    1.9
    """
    code_dictionary = get_codes(tree)
    length_of_code = 0
    count = 0
    for symbol in freq_dict:
        length_of_code += freq_dict[symbol] * len(code_dictionary[symbol])
        count += freq_dict[symbol]
    return length_of_code / count


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']

    """
    string_of_codes = ''
    index = 0
    index_2 = 8
    result_list = []
    for symbol in text:
        string_of_codes += codes[symbol]
    while index <= len(string_of_codes):
        result_list.append(bits_to_byte(string_of_codes[index:index_2]))
        index += 8
        index_2 += 8
    return bytes(result_list)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    nodes_in_post_order = create_post_list(tree)
    final_list = []
    for node in nodes_in_post_order:
        node_children = []
        if node.left.is_leaf():
            node_children.append(0)
            node_children.append(node.left.symbol)
        else:
            node_children.append(1)
            node_children.append(node.left.number)

        if node.right.is_leaf():
            node_children.append(0)
            node_children.append(node.right.symbol)
        else:
            node_children.append(1)
            node_children.append(node.right.number)
        final_list += node_children
    return bytes(final_list)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root = node_lst[root_index]
    root_huffmannode = HuffmanNode(None)
    if root.l_type == 0:
        root_huffmannode.left = HuffmanNode(root.l_data)
    else:
        root_huffmannode.left = generate_tree_general(node_lst, root.l_data)

    if root.r_type == 0:
        root_huffmannode.right = HuffmanNode(root.r_data)
    else:
        root_huffmannode.right = generate_tree_general(node_lst, root.r_data)
    return root_huffmannode


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    >>> l1 = [ReadNode(0, 2, 0, 3), ReadNode(0, 1, 1, 1), ReadNode(0, 4, 0, 5),\
ReadNode(1, 1, 1, 2)]
    >>> generate_tree_postorder(l1, 3)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode\
(None, HuffmanNode(2, None, None), HuffmanNode(3, None, None))), \
HuffmanNode(None, HuffmanNode(4, None, None), HuffmanNode(5, None, None)))
    >>> l2 = [ReadNode(0, 1, 0, 2)]
    >>> generate_tree_postorder(l2, 0)
    HuffmanNode(None, HuffmanNode(1, None, None), HuffmanNode(2, None, None))
    """
    node = HuffmanNode(None)
    root = node_lst[root_index]
    if root.l_type == 0:
        node.left = HuffmanNode(root.l_data)
    else:
        node.left = generate_tree_postorder(node_lst, root_index - 2)
    if root.r_type == 0:
        node.right = HuffmanNode(root.r_data)
    else:
        node.right = generate_tree_postorder(node_lst, root_index - 1)
    return node


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes

    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))), HuffmanNode(None, \
HuffmanNode(4), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> result = generate_uncompressed(t, text, size)
    >>> result == bytes([5, 3, 1, 1])
    True
    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(3), \
HuffmanNode(None, HuffmanNode(1), HuffmanNode(4))), HuffmanNode(None, \
HuffmanNode(2), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> x = generate_uncompressed(t, text, size)
    >>> x == bytes([5, 4, 3, 3])
    True
    >>> t_t = bytes([1, 2, 1, 0])
    >>> d_freq = make_freq_dict(t_t)
    >>> tree = huffman_tree(d_freq)
    >>> d_codes = get_codes(tree)
    >>> compressed = generate_compressed(t_t, d_codes)
    >>> og = t_t
    >>> unc = generate_uncompressed(tree, compressed, len(og))
    >>> og == unc
    True
    """
    bits = ''
    result_bits = []
    for byte in text:
        bits += byte_to_bits(byte)
    index = 0
    cur_node = tree
    while len(result_bits) < size:
        if cur_node.is_leaf():
            result_bits.append(cur_node.symbol)
            cur_node = tree
        elif bits[index] == '0':
            cur_node = cur_node.left
            index += 1
        elif bits[index] == '1':
            cur_node = cur_node.right
            index += 1
    return bytes(result_bits)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> left2 = HuffmanNode(None, HuffmanNode(97), HuffmanNode(98))
    >>> right2 = HuffmanNode(None, HuffmanNode(99), \
    HuffmanNode(None, HuffmanNode(100), HuffmanNode(101)))
    >>> improved = HuffmanNode(None, left2, right2)
    >>> tree == improved
    True
    """
    freq_list = create_sorted_list(freq_dict, get_key_value, True)
    leafs_in_level = level_order_list(tree)
    index = 0
    for level in leafs_in_level:
        if len(level) == 0:
            pass
        else:
            for leaf in level:
                leaf.symbol = freq_list[index][0]
                index += 1


def level_order_list(tree):
    """
    Returns a list of lists where each internal list represents a level of the
     tree and is filled if the leaves on that level if there are any.

    @param HuffmanNode tree: The Root Huffman Node of the Tree to be improved
    @rtype: list[list[HuffmanNodes]]

    >>> t = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> level_order_list(t)
    [[], [HuffmanNode(3, None, None), HuffmanNode(2, None, None)]]
    """
    if tree is None:
        return []
    result = []
    nodes = [tree]
    while nodes:
        result.append([node for node in nodes if node.is_leaf()])
        next_level = []
        for node in nodes:
            if node.left is not None:
                next_level.append(node.left)
            if node.right is not None:
                next_level.append(node.right)
        nodes = next_level

    return result


def create_sorted_list(d, crit, reversal):
    """
    Returns a list[tuple] where the first item in the tuple is the key
    and the second item is the value of the key taken from the dictionary d.
    It also sorts the list based upon the criteria crit (such as what item in
    the tuples to sort by) and the sort order is dependant on reversal, where
    True reverses the list and False does not.

    @param dictionary d: the dictionary to turn into a list
    #do this
    @param function(parameter) crit: the criteria
    @param binary reversal: the Binary value of True or False which states
    whether to reverse the list generated
    """
    result = []
    for key in d:
        result.append((key, d[key],))
    result.sort(key=crit, reverse=reversal)
    return result


if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()
    import time
    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
