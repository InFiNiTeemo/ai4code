def frozen_literal_eval(node_or_string):
    """
    Safely evaluate an expression node or a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None.

    SPECIAL: This version uses frozensets instead of sets
    """
    # SPECIAL: import names from ast module
    from ast import parse, Expression, Str, Bytes, Num, Tuple, List, Set, Dict
    from ast import NameConstant, UnaryOp, UAdd, USub, BinOp, Add, Sub
    # END SPECIAL
    if isinstance(node_or_string, str):
        node_or_string = parse(node_or_string, mode='eval')
    if isinstance(node_or_string, Expression):
        node_or_string = node_or_string.body
    def _convert(node):
        if isinstance(node, (Str, Bytes)):
            return node.s
        elif isinstance(node, Num):
            return node.n
        elif isinstance(node, Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, List):
            return list(map(_convert, node.elts))
        elif isinstance(node, Set):
            #SPECIAL: returns a frozenset
            return frozenset(map(_convert, node.elts))
            # END SPECIAL
        elif isinstance(node, Dict):
            return dict((_convert(k), _convert(v)) for k, v
                        in zip(node.keys, node.values))
        elif isinstance(node, NameConstant):
            return node.value
        elif isinstance(node, UnaryOp) and \
             isinstance(node.op, (UAdd, USub)) and \
             isinstance(node.operand, (Num, UnaryOp, BinOp)):
            operand = _convert(node.operand)
            if isinstance(node.op, UAdd):
                return + operand
            else:
                return - operand
        elif isinstance(node, BinOp) and \
             isinstance(node.op, (Add, Sub)) and \
             isinstance(node.right, (Num, UnaryOp, BinOp)) and \
             isinstance(node.left, (Num, UnaryOp, BinOp)):
            left = _convert(node.left)
            right = _convert(node.right)
            if isinstance(node.op, Add):
                return left + right
            else:
                return left - right
        raise ValueError('malformed node or string: ' + repr(node))
    return _convert(node_or_string)
# fid: code
def get_code_dict(df):
    # print(df['code'][0])
    col = df['code'].apply(lambda x: ' '.join([y[2] for y in frozen_literal_eval(x)]))
    return dict(zip(df['fid'], col))


# fid: gid
def get_group_dict(df):
    return dict(zip(df['fid'], df['gid']))


# to shorter the text,
import re
def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    document = document.replace('_',' ')

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()
    return document
