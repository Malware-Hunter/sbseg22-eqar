# EQAR - ECLAT and Association Rules

These are Python files with implementation of EQAR.

Malware detection through Association Rules generated using the PyFIM - Frequent Item Set Mining for Python library (https://borgelt.net/pyfim.html)

## Environment

The tool has been tested in the following environments:

**Ubuntu 20.04**

- Kernel = `Linux version 5.4.0-120-generic (buildd@lcy02-amd64-006) (gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)) #136-Ubuntu SMP Fri Jun 10 13:40:48 UTC 2022`
- Python = `Python 3.8.10`

### How to use

NOTE: These procedures have only been tested on Ubuntu 20.04

**eqar.py**
- Installing Python requirements:
    ```sh
    $ pip install -r requirements.txt
    ```

- Examples:
    - **EQAR** for the **drebin215.csv** dataset with minimum support at 10%, rule quality **prec** and threshold at 10%
    ```sh
    $ python3 eqar.py -d datasets/drebin215.csv -s 0.1 -c 0.95 -q prec -t 0.1
    ```

- Usage (arguments):
    ```sh
    $ python3 eqar.py --help
    ```
