import mlflow
from mlflow.tracking import MlflowClient
import os
from estimation.config import mlflow_dir_path
from evaluation.mae_discrete_no_aruco import real as real_discrete
from evaluation.mae_real_imgs import real
from estimation.model import efficient_net_b5_discrete

patience = 10


def get_metric_to_mlflow(run_id=None, model_path=None):
    """ Measures the MAE metric for specified run_id and saves it to mlflow. Only implemented for efficientnet_b3 models for now.

    Args:
        run_id: (str, optional): Specifies the run_id of the existing mlflow run. If None it goes through all runs in the mlflow project.
        model_path (str, optional): Specifies path to model in case the mlflow does not contain the model path parameter.
    """
    client = MlflowClient()
    if run_id:
        id_list = [run_id]
    else:
        id_list = os.listdir(mlflow_dir_path)

    for run_id in id_list:
        try:
            run = client.get_run(run_id)

            params = run.data.params

            if "model_path" in params:
                model_path = f"{params['model_path']}/model_{int(params['erpoch_reached']) - patience}"

            if "data_mode" in params:
                if params["data_mode"] == "DataMode.DISCRETE":
                    maeA, maeE = real_discrete(model_path)
                elif params["data_mode"] == "DataMode.RADIANS":
                    maeA, maeE = real(model_path)

                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metric("maeA", maeA)
                    mlflow.log_metric("maeE", maeE)

                    print(f"Run ID: {run_id}")
            else:
                print(f"Parameter 'data_mode' does not exist in run '{run_id}'.")

        except mlflow.exceptions.MlflowException as e:
            print(f"Error retrieving run: {e}")


if __name__ == '__main__':
    #get_metric_to_mlflow("daeeafe852354cf58ddc57f2d70c027f")
    pass