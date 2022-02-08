from javalang.ast import Node

class ASTNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile']:
            return [ASTNode(children[0][1])]
        elif self.token == 'For':
            return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [ASTNode(child) for _, child in children]

class BlockNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def get_token(self, node):
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = 'Modifier'
        elif isinstance(node, Node):
            token = node.__class__.__name__
        else:
            token = ''
        return token

    def ori_children(self, root):
        if isinstance(root, Node):
            if self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
                children = root.children[:-1]
            else:
                children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def add_children(self):
        if self.is_str:
            return []
        logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
        children = self.ori_children(self.node)
        if self.token in logic:
            return [BlockNode(children[0])]
        elif self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
            return [BlockNode(child) for child in children]
        else:
            return [BlockNode(child) for child in children if self.get_token( child) not in logic]

class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token



def get_token(node):
    token = ''
    if isinstance(node,str):
        token = node
    elif isinstance(node,set):
        token = 'Modifier'
    elif isinstance(node,Node):
        token = node.__class__.__name__
    return token

def get_children(root):
    if isinstance(root,Node):
        children = root.children
    elif isinstance(root,set):
        children = list(root)
    else:
        children = []
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item,list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item
    return list(expand(children))

def get_sequence(node, sequence, father, fa, cd):   #SimASTGCN
    token, children = get_token(node), get_children(node)
    if token in ['ForStatement', 'TypeDeclaration', 'ContinueStatement', 'ContinueStatement', 'BreakStatement', 'ThrowStatement','TryStatement','ReturnStatement','WhileStatement', 'DoStatement','SwitchStatement', 'IfStatement', 'MethodDeclaration', 'ConstructorDeclaration', 'ClassDeclaration', 'EnumDeclaration', 'FieldDeclaration', 'VariableDeclarator', 'LocalVariableDeclaration', 'VariableDeclaration', 'FormalParameter'] or token in cd:
        sequence.append(token)
        father.append(fa)
        fa = len(sequence)-1
    for child in children:
        get_sequence(child,sequence,father,fa,cd)
    #if token in ['ForStatement', 'WhileStatement', 'DoStatement','SwitchStatement', 'IfStatement', 'MethodDeclaration', 'ConstructorDeclaration', 'ClassDeclaration', 'EnumDeclaration', 'FieldDeclaration', 'VariableDeclarator', 'LocalVariableDeclaration', 'VariableDeclaration', 'FormalParameter']:
    #    sequence.append('End')



def get_sequences(node, sequence):
    token, children = get_token(node), get_children(node)
    sequence.append(token)
    for child in children:
        get_sequences(child,sequence)


def get_sequencess(node, sequence, father, fa, cd):    #ASTGCN
    token, children = get_token(node), get_children(node)
    sequence.append(token)
    father.append(fa)
    fa = len(sequence)-1
    for child in children:
        get_sequencess(child,sequence,father,fa,cd)