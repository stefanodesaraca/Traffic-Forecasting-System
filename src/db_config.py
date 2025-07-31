from enum import Enum
from src.dbsecrets import superuser

class DBConfig(Enum):
    SUPERUSER = superuser
    SUPERUSER_PASSWORD = ""
    TFS_USER = "tfs"
    TFS_PASSWORD = "tfs"

    HUB_DB = "tfs_hub"
    MAINTENANCE_DB = "postgres"

    DB_HOST = "localhost"




