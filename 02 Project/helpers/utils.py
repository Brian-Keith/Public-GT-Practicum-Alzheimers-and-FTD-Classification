'''Utility functions for the project.

This module contains a variety of utility functions for the project.

Copyright 2024, Brian Keith
All Rights Reserved
'''

from IPython.display import display, Markdown
from subprocess import run
import os
from bs4 import BeautifulSoup as soup
import base64
import datetime as dt

def printmd(string: str):
    '''Takes in a string and displays it as markdown with formatting for the
    headers 1-3. If the string starts with #, it will be interpreted as a 
    header.
    
    Args:
        string (str, required): String to be displayed as markdown starting
            with # for headers 1-3.
    
    Returns:
        None. None is return per PEP 8 recommendations.
    
    Example Usage:
        >>> printmd('# This is a level 1 header')
        >>> printmd('## This is a level 2 header')
        >>> printmd('### This is a level 3 header')
        >>> printmd('This is a normal string in Markdown.')
    '''
    
    header_map = {1:'#B3A369',2:'#003057',3:'#54585A'}
    if string.startswith('#'):
        nh = string.count('#')
        string = string.replace('#','')
        display(Markdown('#'*nh + f' <span style="color:{header_map[nh]}">{string}</span>'))
    else:
        display(Markdown(string))

    return None

def export_code(
    cur_file: str,
    output_dir: str = '', 
    output_name: str = '', 
    cell_tags_exist: bool = False, 
    template: str = 'lab',
    conda_path: str = None,
    conda_env: str = None,
    ):
    '''Export Jupyter Notebook as HTML file

    Args:
        cur_file (str, required): Name of the file function is being used in FULL PATH of the file. Defaults to the name of the ipynb file.
        output_dir (str, optional): Directory to output the file to. Defaults to local directory of Jupyter Notebook.
        output_name (str, optional): Name of the file that will be exported. Defaults to the name of the ipynb file.
        cell_tags_exist (bool, optional): Flag for if cell tags exist . Defaults to False.
        template (str, optional): Template to use for export. Defaults to 'lab'. Options are 'lab' or 'classic'. 'classic' should be used if you're planning to convert the HTML to PDF. 'lab' is better for viewing in browser.
        
    Returns:
        int: 0 if successful, 1 if not
        
    Note:
        If you get an error using this function of `'jupyter' is not recognized as an internal or external command, operable program or batch file`, you may need to add the path to the jupyter executable to your PATH environment variable. You can do this by running the following code in a cell in your Jupyter Notebook:
        
    
    Example Usage:
        >>> export_code(
        ... 'C:\\Users\\USERNAME\\Documents\\Test.ipynb',
        ... 'C:\\Users\\USERNAME\\Documents\\Logs',
        ... 'Test.html',
        ... False,
        ... 'classic'
        ... )
    '''
    def replace_imgs():
        soup_html = soup(open(os.path.join(output_dir, output_name)).read(),features="html.parser")
        img_tags = soup_html.findAll('img')
        img_path = os.path.join(os.path.dirname(output_dir), 'imgs')

        for tag in img_tags:
            #skip any images that already have base64
            if 'base64' in tag['src']:
                continue
            
            img_src = tag['src'].split('/')[-1]
            print(f'Replacing {img_src}')
            tag['src'] = os.path.join(img_path, img_src)
            
            base64_str = base64.b64encode(open(os.path.join(img_path, img_src), 'rb').read()).decode('utf-8')
            new_src = 'data:image/png;base64,' + base64_str
            
            tag['src'] = new_src

        with open(os.path.join(output_dir, output_name), 'w') as f:
            f.write(str(soup_html))
        
        return None
    
    if output_dir == '':
        output_dir = os.getcwd().replace('\\','/')

    if output_name == '':
        cur_file = cur_file.replace('\\', '/')
        output_name = cur_file.split('/')[-1].split('.')[0] + '.html'

    if cell_tags_exist == False:
        process = run([
            'jupyter', 
            'nbconvert',
            "--output-dir={}".format(output_dir),     
            '--to','html',  
            cur_file,
            '--template',f'{template}',
            '--output', f'{output_name}'], 
            shell=True,
            capture_output=True)
    else:
        process = run([
            'jupyter', 
            'nbconvert',
            "--output-dir={}".format(output_dir),     
            '--to','html',
            '--template',f'{template}',
            '--TagRemovePreprocessor.enabled=True',
            '--TagRemovePreprocessor.remove_cell_tags={\"remove_cell\"}',
            '--TagRemovePreprocessor.remove_input_tags={\"remove_input\"}',
            '--no-prompt',
            cur_file,
            '--output', f'{output_name}'], 
            shell=True,
            capture_output=True)
        
    if process.returncode == 0:
        printmd(f'### Code saved to {output_name}')
        replace_imgs()
    else:
        printmd('## REPORT ERROR:')
        import re
        error_str = re.sub(r'\\.',lambda x:{'\\n':'\n','\\t':'\t', '\\r': '\r',"\\'":"'", '\\\\': '\\'}.get(x[0],x[0]),str(process.stderr))
        error_str = error_str.replace('b"','').replace('"','').strip()
        #convert from bytes to string
        print(error_str)
        #! attempts to resolve issue with pathing for jupyter executable
        if "is not recognized as an internal or external command" in error_str:
            printmd('### Attempting to resolve issue with PATH environment variable...')
            if conda_path is None or conda_env is None:
                raise ValueError('Please specify the `conda_path` and `conda_env` arguments to resolve the issue with the PATH environment variable.')
        
            if cell_tags_exist == False:
                process = run([
                    conda_path,
                    conda_env,
                    '&',
                    'jupyter', 
                    'nbconvert',
                    "--output-dir={}".format(output_dir),     
                    '--to','html',  
                    cur_file,
                    '--template',f'{template}',
                    '--output', f'{output_name}'], 
                    shell=True,
                    capture_output=True)
            else:
                process = run([
                    conda_path,
                    conda_env,
                    '&',
                    'jupyter', 
                    'nbconvert',
                    "--output-dir={}".format(output_dir),     
                    '--to','html',
                    '--template',f'{template}',
                    '--TagRemovePreprocessor.enabled=True',
                    '--TagRemovePreprocessor.remove_cell_tags={\"remove_cell\"}',
                    '--TagRemovePreprocessor.remove_input_tags={\"remove_input\"}',
                    '--no-prompt',
                    cur_file,
                    '--output', f'{output_name}'], 
                    shell=True,
                    capture_output=True)
            
            if process.returncode == 0:
                printmd(f'### Code saved to {output_name}')
                replace_imgs()
            else:
                printmd('## REPORT ERROR:')
                error_str = re.sub(r'\\.',lambda x:{'\\n':'\n','\\t':'\t', '\\r': '\r',"\\'":"'", '\\\\': '\\'}.get(x[0],x[0]),str(process.stderr))
                error_str = error_str.replace('b"','').replace('"','').strip()
                #convert from bytes to string
                print(error_str)
                return process.returncode
        
    
    return process.returncode

def format_timing(seconds_delta):
    '''Simple function to return a string of the time in [H]H:MM:SS[.UUUUUU] format.'''
    return str(dt.timedelta(seconds = seconds_delta))
