from enum import Enum
from dbsecrets import superuser, superuser_password

class DBConfig(Enum):
    SUPERUSER = superuser
    SUPERUSER_PASSWORD = superuser_password
    TFS_USER = "tfs"
    TFS_PASSWORD = "tfs"
    TFS_ROLE = "tfs"
    TFS_ROLE_PASSWORD = "tfs"

    HUB_DB = "tfs_hub"
    MAINTENANCE_DB = "postgres"

    DB_HOST = "localhost"




