from prefect import serve

from pipeline_prefect import all_flow, train_flow


if __name__ == "__main__":

    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 9 * * *"
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            cron="0 10 * * *"
        )
    )
