import sys
import linecache
import argparse
import ast
import os
import importlib
import traceback
import contextlib
import multiprocessing
import types
import builtins

class VariableVisitor(ast.NodeVisitor):
    def __init__(self, all_variables=None):
        self.variables = set()
        self.all_variables = all_variables or {}

    def visit_Name(self, node):
        self.variables.add(node.id)

    def visit_Call(self, node):
        #print(f"visiting Call: {ast.dump(node)}")

        # Fix: Handle ast.Name nodes within ast.Call nodes
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
        elif isinstance(node.func, ast.Name):
            self.visit_Name(node.func)

        #print(f"args: {node.args}")  # Debugging

        for arg in node.args:
            #print(f"arg: {ast.dump(arg)}")

            self.visit(arg)

        for keyword in node.keywords:
            #print(f"Visiting keyword: {keyword}")  # Debugging
            self.visit(keyword.value)

    def visit_Assign(self, node):
        # ignore left side of assignment
        self.visit(node.value)


    def visit_Attribute(self, node):  # Modify this method to exclude methods
        attr_name_parts = []

        def collect_attr_parts(node):
            if isinstance(node, ast.Attribute):
                attr_name_parts.append(node.attr)
                collect_attr_parts(node.value)
            elif isinstance(node, ast.Name):
                attr_name_parts.append(node.id)

        collect_attr_parts(node)
        attr_name_parts.reverse()
        attr_full_name = ".".join(attr_name_parts)

        if self.all_variables.get(attr_full_name) is not None:
            if not callable(self.all_variables[attr_full_name]):
                self.variables.add(attr_full_name)

def get_line_variables(line):
    try:
        tree = ast.parse(line)
        visitor = VariableVisitor()
        visitor.visit(tree)
        return visitor.variables
    except SyntaxError:
        return set()
        
def should_print(line):

    try:
        tree = ast.parse(line)
                
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, 
                ast.Assert, ast.Compare, ast.Return, ast.Call)):
                return True
    except SyntaxError:
        pass
    return False
    
def val_str(my_string):
    
    my_string = my_string.strip().replace('\n', ' ')
    
    if len(my_string) <= 50:
        result = my_string
    else:
        first_25_chars = my_string[:25]
        last_25_chars = my_string[-25:]
        result = first_25_chars + "..." + last_25_chars
        
    return result

class Tracer:
    def __init__(self, basename, maxchars=2000, print_output=False):
        self.basename = basename
        self.print_output = print_output
        self.maxchars = maxchars

        self.line_to_string = {}

        self.line_print_order = []

    def get_output(self, save_filename=None):
        # print from end

        remaining = self.maxchars
        lines = []

        while self.line_print_order:
            linenum = self.line_print_order.pop()
            string = self.line_to_string[linenum]

            assert isinstance(string, str), f"expected string: {string}"

            remaining -= len(string) + 1

            if remaining < 0:
                break

            lines.append(string)

        lines.reverse()

        output_str = "\n".join(lines)

        if save_filename:
            with open(save_filename, 'w') as f:
                f.write(output_str)

        if self.print_output:
            print(output_str)
            
        return output_str

    def append(self, linenum, string):
        """append the string with the given line number"""

        self.line_to_string[linenum] = string

        try:
            self.line_print_order.remove(linenum) # may be slower than necessary
        except ValueError:
            pass

        self.line_print_order.append(linenum)

    def trace_function(self, frame, event, _arg):
        if event == "line":
            code = frame.f_code
            line_no = frame.f_lineno
            #func_name = code.co_name

            if os.path.basename(code.co_filename) == self.basename:
                line = linecache.getline(code.co_filename, line_no).strip()
                orig_line = line

                if line[-1] == ":":
                    line += " pass" # prevents sytax errors for ifs or loops when parsing with ast

                if should_print(line):
                    line_vars = get_line_variables(line)
                    
                    # Updated local_vars dictionary construction
                    local_vars = {}
                    for k in line_vars:
                        if "." in k:
                            attr_parts = k.split(".")
                            obj = frame.f_locals.get(attr_parts[0])

                            for i in range(1, len(attr_parts)):
                                if obj is not None:
                                    obj = getattr(obj, attr_parts[i], None)

                            local_vars[k] = obj
                        else:
                            local_vars[k] = frame.f_locals.get(k)
                    
                    try:
                        item_list = []
                        for key, val in sorted(local_vars.items()):
                            # if val is types.ModuleType using a single isinstance:
                            if not isinstance(val, types.ModuleType) and not isinstance(val, type) and not key in builtins.__dict__:
                                item_list.append(f"{key}={val_str(str(val))}")
                                
                        items = ", ".join(item_list)
                    except AttributeError:
                        items = ""
                    
                    if items:
                        self.append(line_no, f"line {line_no:3d}: '{orig_line}', variables before line executes: {items}")

        return self.trace_function    

def trace_worker(module, module_args, maxchars, print_output, return_dict, redirect_stdout):
    # worker for multiprocessing to be able to handle timeouts
    # returns result in return_dict

    if module_args is None:
        module_args = []

    has_error = False
    tb_string = ""
    dt_string = ""

    filename = f"{module.replace('.', '/')}.py"
    basename = os.path.basename(filename)

    tracer = Tracer(basename, maxchars=maxchars, print_output=print_output)
    sys.settrace(tracer.trace_function)
    
    try:    
        loaded = importlib.import_module(module)
       
        if redirect_stdout:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    loaded.main(*module_args)
        else:
            loaded.main(*module_args)
    except Exception as e:
        has_error = True
        tb_string = traceback.format_exc().strip()
    finally:
        sys.settrace(None)

    dt_string = tracer.get_output()

    return_dict['has_error'] = has_error
    return_dict['tb_string'] = tb_string
    return_dict['dt_string'] = dt_string
    
def trace_main(module, module_args=None, maxchars=2000, print_output=False, timeout=None, redirect_stdout=True):
    # returns has_error, traceback_string, dynamic_trace_string

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=trace_worker, args=(module, module_args, maxchars, print_output, return_dict, redirect_stdout))
    p.start()

    p.join(timeout)

    if p.is_alive():
        p.kill()
        p.join()

        return_dict = {}
        return_dict['has_error'] = True
        return_dict['dt_string'] = ''
        return_dict['tb_string'] = f"Error: Timeout of {timeout} seconds exceeded. Code did not exit in time. Modify code to reduce execution time."

    has_error = return_dict['has_error']
    tb_string = return_dict['tb_string']
    dt_string = return_dict['dt_string']

    # not sure how this works with multithreading if we killed things, better to reset it in case of timeouts
    sys.settrace(None)

    return has_error, tb_string, dt_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace an execution printing values to a file')
    parser.add_argument('--module', help='module to import that contains main function', required=True)

    # get multiprocessing argument
    parser.add_argument('--show_stdout', help='run single threaded', action='store_true')
    args = parser.parse_args()

    trace_main(args.module, print_output=True, redirect_stdout=not args.show_stdout)
    
