from tfs_utils import (GlobalDefinitions,
                       GeneralPurposeToolbox,
                       ProjectsHub,
                       ProjectsHubMetadataManager,
                       ProjectMetadataManager,
                       TRPMetadataManager,
                       TRPToolbox,
                       ForecastingToolbox)

gp_toolbox = GeneralPurposeToolbox()
pjh = ProjectsHub()

pjhmm = ProjectsHubMetadataManager(path=pjh.hub / GlobalDefinitions.PROJECTS_HUB_METADATA.value)
pmm = ProjectMetadataManager(path=pjh.hub / pjh.get_current_project() / GlobalDefinitions.PROJECT_METADATA.value)
tmm = TRPMetadataManager() #Defining path is not necessary for TRPMetadataManager

trp_toolbox = TRPToolbox(tmm=tmm)
forecasting_toolbox = ForecastingToolbox(gp_toolbox=gp_toolbox, tmm=tmm, pmm=pmm)
