from enum import Enum

class DBConfig(Enum):
    SUPERUSER = "postgres"
    SUPERUSER_PASSWORD = ""
    TFS_USER = "tfs"
    TFS_PASSWORD = "tfs"

    HUB_DB = "tfs_hub"
    MAINTENANCE_DB = "postgres"

    DB_HOST = "localhost"




