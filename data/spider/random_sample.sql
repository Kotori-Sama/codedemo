[
    {
        "query": "SELECT avg(LifeExpectancy) FROM country WHERE Continent = \"Africa\" AND GovernmentForm = \"Republic\"",
        "db_id": "world_1",
        "question": "What is the average life expectancy in African countries that are republics?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        2
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        5,
                        [
                            0,
                            [
                                0,
                                15,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            10,
                            false
                        ],
                        null
                    ],
                    "\"Africa\"",
                    null
                ],
                "and",
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            19,
                            false
                        ],
                        null
                    ],
                    "\"Republic\"",
                    null
                ]
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT Maker , Model FROM MODEL_LIST;",
        "db_id": "car_1",
        "question": "What are all the makers and models?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        3
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                11,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                12,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT count(*) FROM country WHERE GovernmentForm = \"Republic\"",
        "db_id": "world_1",
        "question": "How many countries have governments that are republics?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        2
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        3,
                        [
                            0,
                            [
                                0,
                                0,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            19,
                            false
                        ],
                        null
                    ],
                    "\"Republic\"",
                    null
                ]
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT T1.Id , T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker GROUP BY T1.Id HAVING count(*) >= 2 INTERSECT SELECT T1.Id , T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id = T2.Maker JOIN CAR_NAMES AS T3 ON T2.model = T3.model GROUP BY T1.Id HAVING count(*) > 3;",
        "db_id": "car_1",
        "question": "What are the ids and makers of all car makers that produce at least 2 models and make more than 3 cars?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        2
                    ],
                    [
                        "table_unit",
                        3
                    ]
                ],
                "conds": [
                    [
                        false,
                        2,
                        [
                            0,
                            [
                                0,
                                6,
                                false
                            ],
                            null
                        ],
                        [
                            0,
                            11,
                            false
                        ],
                        null
                    ]
                ]
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                6,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                7,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [
                [
                    0,
                    6,
                    false
                ]
            ],
            "having": [
                [
                    false,
                    5,
                    [
                        0,
                        [
                            3,
                            0,
                            false
                        ],
                        null
                    ],
                    2.0,
                    null
                ]
            ],
            "orderBy": [],
            "limit": null,
            "intersect": {
                "from": {
                    "table_units": [
                        [
                            "table_unit",
                            2
                        ],
                        [
                            "table_unit",
                            3
                        ],
                        [
                            "table_unit",
                            4
                        ]
                    ],
                    "conds": [
                        [
                            false,
                            2,
                            [
                                0,
                                [
                                    0,
                                    6,
                                    false
                                ],
                                null
                            ],
                            [
                                0,
                                11,
                                false
                            ],
                            null
                        ],
                        "and",
                        [
                            false,
                            2,
                            [
                                0,
                                [
                                    0,
                                    12,
                                    false
                                ],
                                null
                            ],
                            [
                                0,
                                14,
                                false
                            ],
                            null
                        ]
                    ]
                },
                "select": [
                    false,
                    [
                        [
                            0,
                            [
                                0,
                                [
                                    0,
                                    6,
                                    false
                                ],
                                null
                            ]
                        ],
                        [
                            0,
                            [
                                0,
                                [
                                    0,
                                    7,
                                    false
                                ],
                                null
                            ]
                        ]
                    ]
                ],
                "where": [],
                "groupBy": [
                    [
                        0,
                        6,
                        false
                    ]
                ],
                "having": [
                    [
                        false,
                        3,
                        [
                            0,
                            [
                                3,
                                0,
                                false
                            ],
                            null
                        ],
                        3.0,
                        null
                    ]
                ],
                "orderBy": [],
                "limit": null,
                "intersect": null,
                "union": null,
                "except": null
            },
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT Name , SurfaceArea , IndepYear FROM country ORDER BY Population LIMIT 1",
        "db_id": "world_1",
        "question": "Give the name, year of independence, and surface area of the country that has the lowest population.",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        2
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                9,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                12,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                13,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [],
            "having": [],
            "orderBy": [
                "asc",
                [
                    [
                        0,
                        [
                            0,
                            14,
                            false
                        ],
                        null
                    ]
                ]
            ],
            "limit": 1,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT Record_Company FROM orchestra WHERE Year_of_Founded < 2003 INTERSECT SELECT Record_Company FROM orchestra WHERE Year_of_Founded > 2003",
        "db_id": "orchestra",
        "question": "Show the record companies shared by orchestras founded before 2003 and after 2003.",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        1
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                9,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [
                [
                    false,
                    4,
                    [
                        0,
                        [
                            0,
                            10,
                            false
                        ],
                        null
                    ],
                    2003.0,
                    null
                ]
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": {
                "from": {
                    "table_units": [
                        [
                            "table_unit",
                            1
                        ]
                    ],
                    "conds": []
                },
                "select": [
                    false,
                    [
                        [
                            0,
                            [
                                0,
                                [
                                    0,
                                    9,
                                    false
                                ],
                                null
                            ]
                        ]
                    ]
                ],
                "where": [
                    [
                        false,
                        3,
                        [
                            0,
                            [
                                0,
                                10,
                                false
                            ],
                            null
                        ],
                        2003.0,
                        null
                    ]
                ],
                "groupBy": [],
                "having": [],
                "orderBy": [],
                "limit": null,
                "intersect": null,
                "union": null,
                "except": null
            },
            "union": null,
            "except": null
        }
    },
    {
        "query": "select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode join city as t3 on t1.code = t3.countrycode where t2.isofficial = 't' and t2.language = 'chinese' and t1.continent = \"asia\"",
        "db_id": "world_1",
        "question": "Which unique cities are in Asian countries where Chinese is the official language ?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        2
                    ],
                    [
                        "table_unit",
                        3
                    ],
                    [
                        "table_unit",
                        0
                    ]
                ],
                "conds": [
                    [
                        false,
                        2,
                        [
                            0,
                            [
                                0,
                                8,
                                false
                            ],
                            null
                        ],
                        [
                            0,
                            23,
                            false
                        ],
                        null
                    ],
                    "and",
                    [
                        false,
                        2,
                        [
                            0,
                            [
                                0,
                                8,
                                false
                            ],
                            null
                        ],
                        [
                            0,
                            3,
                            false
                        ],
                        null
                    ]
                ]
            },
            "select": [
                true,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                2,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            25,
                            false
                        ],
                        null
                    ],
                    "\"t\"",
                    null
                ],
                "and",
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            24,
                            false
                        ],
                        null
                    ],
                    "\"chinese\"",
                    null
                ],
                "and",
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            10,
                            false
                        ],
                        null
                    ],
                    "\"asia\"",
                    null
                ]
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT winner_name , winner_rank_points FROM matches GROUP BY winner_name ORDER BY count(*) DESC LIMIT 1",
        "db_id": "wta_1",
        "question": "What is the name of the winner who has won the most matches, and how many rank points does this player have?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        1
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                34,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                36,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [
                [
                    0,
                    34,
                    false
                ]
            ],
            "having": [],
            "orderBy": [
                "desc",
                [
                    [
                        0,
                        [
                            3,
                            0,
                            false
                        ],
                        null
                    ]
                ]
            ],
            "limit": 1,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT T1.course_name , T1.course_id FROM Courses AS T1 JOIN Sections AS T2 ON T1.course_id = T2.course_id GROUP BY T1.course_id HAVING count(*) <= 2",
        "db_id": "student_transcripts_tracking",
        "question": "What are the names and ids of every course with less than 2 sections?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        1
                    ],
                    [
                        "table_unit",
                        4
                    ]
                ],
                "conds": [
                    [
                        false,
                        2,
                        [
                            0,
                            [
                                0,
                                10,
                                false
                            ],
                            null
                        ],
                        [
                            0,
                            24,
                            false
                        ],
                        null
                    ]
                ]
            },
            "select": [
                false,
                [
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                11,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        0,
                        [
                            0,
                            [
                                0,
                                10,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [],
            "groupBy": [
                [
                    0,
                    10,
                    false
                ]
            ],
            "having": [
                [
                    false,
                    6,
                    [
                        0,
                        [
                            3,
                            0,
                            false
                        ],
                        null
                    ],
                    2.0,
                    null
                ]
            ],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    },
    {
        "query": "SELECT count(*) , max(Percentage) FROM countrylanguage WHERE LANGUAGE = \"Spanish\" GROUP BY CountryCode",
        "db_id": "world_1",
        "question": "What is the total number of countries where Spanish is spoken by the largest percentage of people?",
        "sql": {
            "from": {
                "table_units": [
                    [
                        "table_unit",
                        3
                    ]
                ],
                "conds": []
            },
            "select": [
                false,
                [
                    [
                        3,
                        [
                            0,
                            [
                                0,
                                0,
                                false
                            ],
                            null
                        ]
                    ],
                    [
                        1,
                        [
                            0,
                            [
                                0,
                                26,
                                false
                            ],
                            null
                        ]
                    ]
                ]
            ],
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            24,
                            false
                        ],
                        null
                    ],
                    "\"Spanish\"",
                    null
                ]
            ],
            "groupBy": [
                [
                    0,
                    23,
                    false
                ]
            ],
            "having": [],
            "orderBy": [],
            "limit": null,
            "intersect": null,
            "union": null,
            "except": null
        }
    }
]