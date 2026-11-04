import nanoobjectdetection_legacy as nd

if __name__=="__main__":
    # check python version of env. if not 3.9, raise an error
    import sys
    print(sys.version)
    if sys.version_info[:2] != (3, 9):
        raise EnvironmentError("Python 3.9 is required please choose the py39 environment for the legacy code.")
    nd.Start.NewEvaluation()