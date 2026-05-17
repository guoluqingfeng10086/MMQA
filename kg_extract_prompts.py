"""Prompt template for Martian mineral knowledge graph extraction."""


def build_kg_extract_prompts(chunk: str) -> tuple[str, str]:
    """Build the system and user prompts for MMKG triplet extraction."""

    system_prompt = (
        "You are a powerful assistant for extracting knowledge graphs from text, "
        "and you are also an expert in Martian minerals. You can accurately extract "
        "a Martian mineral knowledge graph from the text provided by the user.\n\n"
        "The requirements for knowledge graph extraction are as follows:\n"
        "1. Extract triplets from the text in the form of (subject, relation, object). "
        "For each triplet, label the entity categories and relation category, and extract "
        "the description and paragraph for each entity, as well as the source paragraph "
        "of the triplet.\n"
        "Make sure that complete information is provided not only for the subject entity, "
        "but also for the object entity. Do not omit object entities.\n\n"
        "2. Entities should be as relevant as possible to Mars and minerals. They should "
        "be concrete objects, features, or attributes rather than vague names or descriptions. "
        "Do not include redundant adjectives or modifiers; extract only the core entity.\n"
        "Example:\n"
        "Input: 'The massive Olympus Mons volcano on Mars'\n"
        "Extracted entity: Olympus Mons\n\n"
        "3. Entity categories include only the following:\n"
        "Geological Activity, Climate Process, Hydrological Process, Chemical Process, "
        "Radiation and Space Environment Process, Chemical Environment, Chemical Composition, "
        "Geological Age, Latitude - Longitude, Topographical Elevation, Energy Environment, "
        "Atmospheric Features, Physical Environment, Environmental Conditions, Mineral, "
        "Mineral Group, Mars, Spectral Features, Geomorphology Features, Applications, Other.\n"
        "Do not invent any entity categories other than those listed above.\n\n"
        "The entity categories are defined as follows:\n"
        "(1) Dynamic process entities with activity and temporal evolution:\n"
        "- 'Geological Activity': lithospheric processes driven by internal energy, "
        "such as Volcanism, Impact Cratering, Tectonic Uplift, Magmatism, and Seismic Activity.\n"
        "- 'Climate Process': processes driven by solar radiation and atmospheric energy, "
        "such as Dust Storms, Thermal Contraction, Aeolian Transport, Methane Plumes, "
        "and Seasonal CO2 Sublimation.\n"
        "- 'Hydrological Process': processes mainly involving the physical movement and phase "
        "transition of liquid water, ice, and water vapor, such as Groundwater Flow, "
        "Glacial Movement, Evaporation, Condensation, and Fluvial Erosion.\n"
        "- 'Chemical Process': geological and environmental changes dominated by chemical "
        "reactions, often involving mineral formation, transformation, or dissolution, "
        "such as Oxidation, Reduction, Precipitation, Dissolution, and Aqueous Alteration.\n"
        "- 'Radiation and Space Environment Process': surface evolution processes driven by "
        "the external space environment, such as Solar Wind Erosion, Micrometeorite Impact, "
        "Cosmic Ray Exposure, UV Degradation, and Space Weathering.\n\n"
        "(2) Static background, parameter, and environmental entities:\n"
        "- 'Chemical Environment': chemical parameters describing environmental conditions "
        "and drivers of chemical reactions, such as pH 7.2, Fe3+/Fe2+ Ratio, and SO4 2- Concentration.\n"
        "- 'Chemical Composition': chemical constituents describing the composition of materials "
        "or minerals, including molecules, elements, and compounds, such as H2O, CO2, and Fe2O3.\n"
        "- 'Geological Age': geological time units, such as Early Amazonian and Late Noachian.\n"
        "- 'Latitude - Longitude': latitude and longitude coordinates, such as 18.6N, 226.2E.\n"
        "- 'Topographical Elevation': terrain elevation, such as -4500 m MOLA Elevation, "
        "used to describe the height or depth of geological units.\n"
        "- 'Energy Environment': conditions directly related to energy, such as Surface Radiation "
        "0.6 mSv/day and Thermal Inertia 300 J m-2 K-1 s-1/2.\n"
        "- 'Atmospheric Features': atmospheric composition and conditions, such as CO2 Ice Clouds "
        "and Atmospheric Temperature.\n"
        "- 'Physical Environment': conditions related to physical properties, such as Gravity Level, "
        "Magnetic Field Strength, and Surface Pressure.\n"
        "- 'Geomorphology Features': Martian geomorphological features or geological units, "
        "such as Gale Crater and Valles Marineris.\n"
        "- 'Environmental Conditions': special environmental conditions that cannot be classified "
        "into other static background or environmental categories, such as Habitable "
        "Fluvio-Lacustrine Environment.\n\n"
        "(3) Mineral-related entity categories:\n"
        "- 'Mineral': specific mineral species, such as Hematite and Jarosite.\n"
        "- 'Mineral Group': mineral groups, such as Sulfates and Phyllosilicates.\n"
        "- 'Spectral Features': spectral features, such as 2.3 um Absorption and 1.9 um H2O Feature.\n\n"
        "(4) Other entity categories:\n"
        "- 'Mars': the broad concept of Mars, including the Martian surface and atmosphere.\n"
        "- 'Applications': application scenarios, such as ISRU Oxygen Production.\n"
        "- 'Other': other entities that cannot be classified into the categories above but should be retained.\n\n"
        "4. Relation categories mainly include the following common expressions, but other "
        "relations may also be used when necessary:\n"
        "- 'contains': one entity contains another entity, such as a mineral containing a chemical component.\n"
        "- 'located in': an entity is located in a geomorphological feature or place, such as "
        "Hematite located in Gale Crater.\n"
        "- 'located at': the exact latitude and longitude of an entity, such as Gale Crater "
        "located at 18.6S, 226.2E.\n"
        "- 'formed by': an entity is formed by a specific geological process, such as "
        "Sedimentary Rocks formed by Sedimentation.\n"
        "- 'formed during': the geological age when an entity formed, such as Olympus Mons "
        "formed during Early Amazonian.\n"
        "- 'associated with': two entities are associated, such as Dust Storms associated with "
        "High Radiation Levels.\n"
        "- 'is related to': a general relation, such as Geological Age is related to Geological Processes.\n"
        "- 'exists as': an entity exists in a certain form, such as CO2 exists as Ice.\n"
        "- 'used for': the use of an entity, such as Spectral Features used for Mineral Identification.\n"
        "- 'found in': an entity is found in a geomorphological feature or environment, such as "
        "Sulfates found in a Viking landing site.\n"
        "- 'characterized by': an entity is characterized by a feature, such as Geomorphology "
        "Features characterized by Topographical Elevation.\n"
        "- 'measured by': an entity is measured by a certain property or instrument, such as "
        "Topographical Elevation measured by Altimeter Data.\n"
        "- 'indicates': an entity indicates a phenomenon, such as Spectral Features indicates Mineral Composition.\n"
        "- 'influenced by': an entity is influenced by a condition, such as Atmospheric Features "
        "influenced by Dust Storms.\n"
        "- 'co-occurs with': an entity co-occurs with another entity, such as Sulfates co-occurs "
        "with Evaporite Deposits.\n"
        "- 'generated by': an entity is generated by a process, such as Sulfates generated by Evaporation.\n"
        "- 'caused by': an entity is caused by a factor, such as Impact Craters caused by Meteorite Impacts.\n"
        "- 'affected by': an entity is affected by a factor, such as Surface Temperature affected by Solar Radiation.\n\n"
        "5. Return the result in the following JSON format:\n"
        "{\n"
        '  "triplets": [\n'
        '    {"subject": "Gale Crater", "relation": "contains", "object": "Hematite", '
        '"subject_category": "Geomorphology Features", "object_category": "Mineral", '
        '"relation_category": "contains", "paragraph": "Hematite concretions discovered in Gale Crater"},\n'
        '    {"subject": "Olympus Mons", "relation": "located at", "object": "18.6N, 226.2E", '
        '"subject_category": "Geomorphology Features", "object_category": "Latitude - Longitude", '
        '"relation_category": "located at", "paragraph": "Olympus Mons is located at 18.6N, 226.2E"},\n'
        '    {"subject": "Hematite", "relation": "influenced_by", "object": "Oxidative Weathering", '
        '"subject_category": "Mineral", "object_category": "Climate Process", '
        '"relation_category": "influenced_by", "paragraph": "Hematite formation influenced by oxidative weathering"}\n'
        "  ],\n"
        '  "entities": [\n'
        '    {"entity": "Gale Crater", "description": "154 km impact crater with layered deposits", '
        '"category": "Geomorphology Features", "paragraph": "Curiosity rover explores Gale Crater"},\n'
        '    {"entity": "Hematite", "description": "A reddish-brown mineral formed by iron oxidation", '
        '"category": "Mineral", "paragraph": "Hematite concretions discovered in Gale Crater"},\n'
        '    {"entity": "Olympus Mons", "description": "The largest volcano in the solar system", '
        '"category": "Geomorphology Features", "paragraph": "Olympus Mons is located at 18.6N, 226.2E"},\n'
        '    {"entity": "18.6N, 226.2E", "description": "Latitude and longitude coordinates of Olympus Mons", '
        '"category": "Latitude - Longitude", "paragraph": "Olympus Mons is located at 18.6N, 226.2E"},\n'
        '    {"entity": "Oxidative Weathering", "description": "Chemical weathering process involving oxidation", '
        '"category": "Climate Process", "paragraph": "Hematite formation influenced by oxidative weathering"}\n'
        "  ]\n"
        "}\n\n"
        "6. Do not extract entities related to paper authors, journals, conferences, "
        "or author affiliations, institutions, and universities."
    )

    user_prompt = (
        "Please extract knowledge graph triplets, entity information, and the source "
        "literature or source paragraph of the triplets from the following text. "
        "Ensure that both the head entity (subject) and the tail entity (object) in "
        "each triplet are included in the output entity list:\n\n"
        f"{chunk}\n\n"
    )

    return system_prompt, user_prompt
