import os

def select_model(model_zoo_path="/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/SemanticSegmentation/models"):
    """
        print available models
        return ( str(id) , model_name )
    """
    models_list=os.listdir(model_zoo_path)
    for id,(model_name) in enumerate(models_list):
        print(id,model_name)
    id = input("Input id to select a model: ")
    try:
        if not (0 <= eval(id) < len(models_list)):
            raise ValueError("Invalid input")
    except:
        raise ValueError("Invalid input")
    model_name=models_list[eval(id)]

    return (id,model_name)