from __future__ import annotations


from eval.benchmarks.agent_abstain.data.environments.agriculture_and_yield.environment import AgricultureAndYieldEnvironment
from eval.benchmarks.agent_abstain.data.environments.astrology_charting.environment import AstrologyChartingEnvironment
from eval.benchmarks.agent_abstain.data.environments.autonomous_delivery_and_routing.environment import AutonomousDeliveryAndRoutingEnvironment
from eval.benchmarks.agent_abstain.data.environments.calendar.environment import CalendarEnvironment
from eval.benchmarks.agent_abstain.data.environments.chat_and_group_management.environment import ChatAndGroupManagementEnvironment
from eval.benchmarks.agent_abstain.data.environments.clinical_records_and_claims.environment import ClinicalRecordsAndClaimsEnvironment
from eval.benchmarks.agent_abstain.data.environments.compliance_and_contracts.environment import ComplianceAndContractsEnvironment
from eval.benchmarks.agent_abstain.data.environments.consumer_banking.environment import ConsumerBankingEnvironment
from eval.benchmarks.agent_abstain.data.environments.consumer_health_portal.environment import ConsumerHealthPortalEnvironment
from eval.benchmarks.agent_abstain.data.environments.crm_and_company_lookup.environment import CrmAndCompanyLookupEnvironment
from eval.benchmarks.agent_abstain.data.environments.device_privacy_and_focus.environment import DevicePrivacyAndFocusEnvironment
from eval.benchmarks.agent_abstain.data.environments.disaster_relief_operations.environment import DisasterReliefOperationsEnvironment
from eval.benchmarks.agent_abstain.data.environments.document_authoring_and_publication.environment import DocumentAuthoringAndPublicationEnvironment
from eval.benchmarks.agent_abstain.data.environments.documents_and_analysis.environment import DocumentsAndAnalysisEnvironment
from eval.benchmarks.agent_abstain.data.environments.education_and_campus_portals.environment import EducationAndCampusPortalsEnvironment
from eval.benchmarks.agent_abstain.data.environments.filesystem.environment import FilesystemEnvironment
from eval.benchmarks.agent_abstain.data.environments.fitness_and_wellness_logs.environment import FitnessAndWellnessLogsEnvironment
from eval.benchmarks.agent_abstain.data.environments.flight_and_travel_management.environment import FlightAndTravelManagementEnvironment
from eval.benchmarks.agent_abstain.data.environments.gmail_and_email_records.environment import GmailAndEmailRecordsEnvironment
from eval.benchmarks.agent_abstain.data.environments.home_medication_inventory.environment import HomeMedicationInventoryEnvironment
from eval.benchmarks.agent_abstain.data.environments.identity_credit_and_collections.environment import IdentityCreditAndCollectionsEnvironment
from eval.benchmarks.agent_abstain.data.environments.industrial_and_infrastructure_control.environment import IndustrialAndInfrastructureControlEnvironment
from eval.benchmarks.agent_abstain.data.environments.maps_and_navigation.environment import MapsAndNavigationEnvironment
from eval.benchmarks.agent_abstain.data.environments.metrics_and_spreadsheet_analysis.environment import MetricsAndSpreadsheetAnalysisEnvironment
from eval.benchmarks.agent_abstain.data.environments.notes_and_reference.environment import NotesAndReferenceEnvironment
from eval.benchmarks.agent_abstain.data.environments.personal_profile_and_contacts.environment import PersonalProfileAndContactsEnvironment
from eval.benchmarks.agent_abstain.data.environments.phone_and_messages.environment import PhoneAndMessagesEnvironment
from eval.benchmarks.agent_abstain.data.environments.project_logs.environment import ProjectLogsEnvironment
from eval.benchmarks.agent_abstain.data.environments.retail_orders.environment import RetailOrdersEnvironment
from eval.benchmarks.agent_abstain.data.environments.science_and_environment_data.environment import ScienceAndEnvironmentDataEnvironment
from eval.benchmarks.agent_abstain.data.environments.security_and_privacy_admin.environment import SecurityAndPrivacyAdminEnvironment
from eval.benchmarks.agent_abstain.data.environments.smart_home.environment import SmartHomeEnvironment
from eval.benchmarks.agent_abstain.data.environments.social_media_dataset_analysis.environment import SocialMediaDatasetAnalysisEnvironment
from eval.benchmarks.agent_abstain.data.environments.spotify.environment import SpotifyEnvironment
from eval.benchmarks.agent_abstain.data.environments.store_procurement_and_inventory.environment import StoreProcurementAndInventoryEnvironment
from eval.benchmarks.agent_abstain.data.environments.system_operations.environment import SystemOperationsEnvironment
from eval.benchmarks.agent_abstain.data.environments.trading_and_portfolio.environment import TradingAndPortfolioEnvironment
from eval.benchmarks.agent_abstain.data.environments.vehicle_status_and_control.environment import VehicleStatusAndControlEnvironment
from eval.benchmarks.agent_abstain.data.environments.venmo_and_shared_expenses.environment import VenmoAndSharedExpensesEnvironment
from eval.benchmarks.agent_abstain.data.environments.weather_and_alerts.environment import WeatherAndAlertsEnvironment
from eval.benchmarks.agent_abstain.data.environments.web_and_cms.environment import WebAndCmsEnvironment
from eval.benchmarks.agent_abstain.data.environments.workforce_and_hr.environment import WorkforceAndHrEnvironment


ENVIRONMENT_REGISTRY = {
    "agriculture_and_yield": AgricultureAndYieldEnvironment,
    "astrology_charting": AstrologyChartingEnvironment,
    "autonomous_delivery_and_routing": AutonomousDeliveryAndRoutingEnvironment,
    "calendar": CalendarEnvironment,
    "chat_and_group_management": ChatAndGroupManagementEnvironment,
    "clinical_records_and_claims": ClinicalRecordsAndClaimsEnvironment,
    "compliance_and_contracts": ComplianceAndContractsEnvironment,
    "consumer_banking": ConsumerBankingEnvironment,
    "consumer_health_portal": ConsumerHealthPortalEnvironment,
    "crm_and_company_lookup": CrmAndCompanyLookupEnvironment,
    "device_privacy_and_focus": DevicePrivacyAndFocusEnvironment,
    "disaster_relief_operations": DisasterReliefOperationsEnvironment,
    "document_authoring_and_publication": DocumentAuthoringAndPublicationEnvironment,
    "documents_and_analysis": DocumentsAndAnalysisEnvironment,
    "education_and_campus_portals": EducationAndCampusPortalsEnvironment,
    "filesystem": FilesystemEnvironment,
    "fitness_and_wellness_logs": FitnessAndWellnessLogsEnvironment,
    "flight_and_travel_management": FlightAndTravelManagementEnvironment,
    "gmail_and_email_records": GmailAndEmailRecordsEnvironment,
    "home_medication_inventory": HomeMedicationInventoryEnvironment,
    "identity_credit_and_collections": IdentityCreditAndCollectionsEnvironment,
    "industrial_and_infrastructure_control": IndustrialAndInfrastructureControlEnvironment,
    "maps_and_navigation": MapsAndNavigationEnvironment,
    "metrics_and_spreadsheet_analysis": MetricsAndSpreadsheetAnalysisEnvironment,
    "notes_and_reference": NotesAndReferenceEnvironment,
    "personal_profile_and_contacts": PersonalProfileAndContactsEnvironment,
    "phone_and_messages": PhoneAndMessagesEnvironment,
    "project_logs": ProjectLogsEnvironment,
    "retail_orders": RetailOrdersEnvironment,
    "science_and_environment_data": ScienceAndEnvironmentDataEnvironment,
    "security_and_privacy_admin": SecurityAndPrivacyAdminEnvironment,
    "smart_home": SmartHomeEnvironment,
    "social_media_dataset_analysis": SocialMediaDatasetAnalysisEnvironment,
    "spotify": SpotifyEnvironment,
    "store_procurement_and_inventory": StoreProcurementAndInventoryEnvironment,
    "system_operations": SystemOperationsEnvironment,
    "trading_and_portfolio": TradingAndPortfolioEnvironment,
    "vehicle_status_and_control": VehicleStatusAndControlEnvironment,
    "venmo_and_shared_expenses": VenmoAndSharedExpensesEnvironment,
    "weather_and_alerts": WeatherAndAlertsEnvironment,
    "web_and_cms": WebAndCmsEnvironment,
    "workforce_and_hr": WorkforceAndHrEnvironment,
}


def get_environment_class(env_type: str):
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_type}'. Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )
    return ENVIRONMENT_REGISTRY[env_type]


def list_environments() -> list[str]:
    return list(ENVIRONMENT_REGISTRY)


def build_multi_environment(env_types, initial_states):
    """Factory for MultiEnvironment. Imported lazily to avoid a cycle during
    registry construction (multi.py imports from this module)."""
    from eval.benchmarks.agent_abstain.data.runtime.multi import build_multi_environment as _build
    return _build(env_types, initial_states)
