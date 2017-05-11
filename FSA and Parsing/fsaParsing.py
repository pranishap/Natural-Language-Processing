# coding: utf-8


import unittest

def run_fsa(states, initial_state, accept_states, transition, input_symbols):
    state = initial_state
    for i in input_symbols:
        if state in transition.keys():
            t = transition[state]
            if i not in t.keys():
                return False
            else:
                state = transition[state][i]
        else:
            return False

    if state in accept_states:
        return True
    else:
        return False



def get_name_fsa():
    states = [0, 1, 2, 3, 4]
    initial_state = 0
    accept_states = [4]
    transition = {
        0: {'Mr.': 1,'Ms.': 1, 'Frank': 2, 'Flo': 2},
        1: {'Frank': 2, 'Flo': 2},
        2: {'Michael':3, 'Maggie':3,'Lewis':4, 'Lutz':4},
        3: {'Lewis':4, 'Lutz':4}
    }
    return states,initial_state,accept_states,transition

def read_grammar(lines):
    tuppleList = []
    for line in lines:
        ls = line.split(':-')
        rhsList = ls[1].split()
        ls[0] = ls[0].strip(' ')
        newRhs = [x.strip(' ') for x in rhsList]
        tuple = (ls[0],newRhs)
        tuppleList.append(tuple)
    return tuppleList



class Tree:
    def __init__(self, label, children=[]):
        self.label = label
        self.children = children

    def __str__(self):
        s = self.label
        for c in self.children:
            s += ' ( ' + str(c) + ' ) '
        return s

    def get_leaves(self):
        leavesList = []
        def get_leaf_nodes( node):
            if node is not None:
                if len(node.children) == 0:
                    leavesList.append(node.label)
                for n in node.children:
                    get_leaf_nodes(n)
        get_leaf_nodes(self)
        return leavesList
    
    

    def get_productions(self):
        prodList = []
        def get_prod(node):
            lhs = node.label
            childList = []
            for c in node.children:
                childList.append(c.label)
            tuple = (lhs,childList)
            prodList.append(tuple)
        
        def get_leaf_nodes( node):
            if node is not None:
                if len(node.children) != 0:
                    get_prod(node)
                for n in node.children:
                    get_leaf_nodes(n)
        get_leaf_nodes(self)
        return prodList



def is_pos(rule, rules):
    lhsList = []
    for l in rules:
        lhsList.append(l[0])
    for rhs in rule[1]:
        if rhs in lhsList:
            return False
    return True


def is_valid_production(production, rules):
    if production in rules:
        return True
    else:
        isPOS = is_pos(production, rules)
        if isPOS:
            r = ''
            for l in rules:
                if production[0] == l[0]:
                    r = l
            if len(r) > 0:
                rRhs = r[1]
                ruleRhs = production[1]
                intersectList = list(set(rRhs) & set(ruleRhs))
                if len(intersectList) != len(ruleRhs):
                    return False
                else:
                        return True
            else:
                return False
        else:
            return False




def is_valid_tree(tree, rules, words):
    leavesList = tree.get_leaves()
    prodList = tree.get_productions()
    for prod in prodList:
        isValidProd = is_valid_production(prod, rules)
        if isValidProd == False:
            return False
    if len(words) == len(leavesList):
        for i in range(len(words)):
            if words[i] != leavesList[i]:
                return False
    else :
        return False
    return True

class TestA1(unittest.TestCase):



    def test_baa_fsa(self):
        print('test')
        states = [0, 1, 2, 3, 4]
        initial_state = 0
        accept_states = [4]
        transition = {
            0: {'b': 1},
            1: {'a': 2},
            2: {'a': 3},
            3: {'a': 3, '!': 4},
            }
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, list('baa!')))
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, list('baaaa!')))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, list('')))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, list('baa')))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, list('bac')))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, list('baaa!a')))

        states, initial_state, accept_states, transition = get_name_fsa()
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, ['Mr.', 'Frank', 'Michael', 'Lewis']))
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, ['Ms.', 'Frank', 'Lewis']))
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, ['Flo', 'Michael', 'Lutz']))
        self.assertTrue(run_fsa(states, initial_state, accept_states, transition, ['Flo', 'Lutz']))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, ['Flo', 'Michael']))
        self.assertFalse(run_fsa(states, initial_state, accept_states, transition, ['Michael']))


        grammar_rules = ['S :- NP VP',
                     'NP :- Det Noun',
                     'NP :- ProperNoun',
                     'VP :- Verb',
                     'VP :- Verb NP',
                     'Det :- a the',
                     'Noun :- book flight',
                     'Verb :- book books include',
                     'ProperNoun :- Houston TWA John']
        rules = read_grammar(grammar_rules)
        rules = sorted(rules)
        self.assertEqual(rules[0][0], 'Det')
        self.assertEqual(rules[0][1][0], 'a')
        self.assertEqual(rules[0][1][1], 'the')
        self.assertEqual(rules[1][0], 'NP')
        self.assertEqual(rules[1][1][0], 'Det')
        self.assertEqual(rules[1][1][1], 'Noun')

        tree = Tree('S', [Tree('NP',
                            [Tree('N', [Tree('John')])]),
                        Tree('VP',
                            [Tree('V', [Tree('books')]),
                                Tree('N', [Tree('flight')])])])
        leaves = tree.get_leaves()
        self.assertListEqual(['John', 'books', 'flight'], leaves)

        tree = Tree('S', [Tree('NP',
                            [Tree('N', [Tree('John')])]),
                        Tree('VP',
                            [Tree('V', [Tree('books')]),
                                Tree('N', [Tree('flight')])])])
                           
        productions = tree.get_productions()
        self.assertEqual(productions[0], ('S', ['NP', 'VP']))
        self.assertEqual(productions[1], ('NP', ['N']))
        self.assertEqual(productions[2], ('N', ['John']))
        self.assertEqual(productions[3], ('VP', ['V', 'N']))
        self.assertEqual(productions[4], ('V', ['books']))
        self.assertEqual(productions[5], ('N', ['flight']))

        rules = [('S', ['NP', 'VP']),
                ('NP', ['ProperNoun']),
                ('ProperNoun', ['John', 'Mary']),
                ('VP', ['V', 'ProperNoun']),
                ('V', ['likes', 'hates'])]
        self.assertFalse(is_pos(('S', ['NP', 'VP']), rules))
        self.assertFalse(is_pos(('NP', ['ProperNoun']), rules))
        self.assertTrue(is_pos(('ProperNoun', ['John', 'Mary']), rules))
        self.assertTrue(is_pos(('V', ['likes', 'hates']), rules))

        rules = [('S', ['NP', 'VP']),
                ('NP', ['ProperNoun']),
                ('ProperNoun', ['John', 'Mary']),
                ('VP', ['V', 'ProperNoun']),
                ('V', ['likes', 'hates'])]
        self.assertTrue(is_valid_production(('S', ['NP', 'VP']), rules))
        self.assertTrue(is_valid_production(('NP', ['ProperNoun']), rules))
        self.assertTrue(is_valid_production(('ProperNoun', ['John']), rules))
        self.assertTrue(is_valid_production(('ProperNoun', ['Mary']), rules))
        self.assertTrue(is_valid_production(('V', ['likes']), rules))
    
        self.assertFalse(is_valid_production(('S', ['VP', 'NP']), rules))
        self.assertFalse(is_valid_production(('V', ['John']), rules))
        self.assertFalse(is_valid_production(('NP', ['NP', 'VP']), rules))

        rules = [('S', ['NP', 'VP']),
                ('NP', ['N']),
                ('NP', ['D', 'N']),
                ('N', ['John', 'flight', 'book']),
                ('D', ['the', 'a']),
                ('VP', ['V', 'NP']),
                ('V', ['books', 'book', 'likes', 'hates']),
                ]
        tree = Tree('S', [Tree('NP', [Tree('N', [Tree('John')])]),
                          Tree('VP', [Tree('V', [Tree('books')]),
                                      Tree('NP', [Tree('N', [Tree('flight')])])])])
        self.assertTrue(is_valid_tree(tree, rules, ['John', 'books', 'flight']))
        self.assertFalse(is_valid_tree(tree, rules, ['John', 'books', 'likes']))

        tree2 = Tree('S', [Tree('NP', [Tree('N', [Tree('John')])]),
                        Tree('VP', [Tree('V', [Tree('books')]),
                                    Tree('NP', [Tree('D', [Tree('the')]),
                                                Tree('N', [Tree('flight')])])])])
        self.assertTrue(is_valid_tree(tree2, rules, ['John', 'books', 'the', 'flight']))
        self.assertFalse(is_valid_tree(tree2, rules, ['John', 'books', 'flight']))

        tree3 = Tree('S', [Tree('NP', [Tree('N', [Tree('John')])]),
                        Tree('VP', [Tree('V', [Tree('books')]),
                                    Tree('NP', [Tree('D', [Tree('flight')])])])])
        self.assertFalse(is_valid_tree(tree3, rules, ['John', 'books', 'flight']))


if __name__ == '__main__':
    unittest.main()






