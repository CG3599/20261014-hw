const EXPERIMENT = {
  "curves": [
    {
      "method": "EFS",
      "count": 1,
      "features": [
        "R&D Spend"
      ],
      "rmse": 7714.334890830185,
      "r2": 0.9265108109341951
    },
    {
      "method": "EFS",
      "count": 2,
      "features": [
        "R&D Spend",
        "Administration"
      ],
      "rmse": 9152.274782350905,
      "r2": 0.8965609365509308
    },
    {
      "method": "EFS",
      "count": 3,
      "features": [
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 8995.90580336142,
      "r2": 0.900065308303732
    },
    {
      "method": "EFS",
      "count": 4,
      "features": [
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9010.31225499439,
      "r2": 0.899744972111299
    },
    {
      "method": "EFS",
      "count": 5,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9055.957323501078,
      "r2": 0.8987266414319106
    },
    {
      "method": "RFECV",
      "count": 1,
      "features": [
        "State_California"
      ],
      "rmse": 35769.73536283392,
      "r2": -0.5800030766990909
    },
    {
      "method": "RFECV",
      "count": 2,
      "features": [
        "State_New York",
        "State_California"
      ],
      "rmse": 35902.08192702272,
      "r2": -0.5917166005474785
    },
    {
      "method": "RFECV",
      "count": 3,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend"
      ],
      "rmse": 7946.372243720277,
      "r2": 0.922023400960793
    },
    {
      "method": "RFECV",
      "count": 4,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration"
      ],
      "rmse": 9316.536408854245,
      "r2": 0.8928146452937172
    },
    {
      "method": "RFECV",
      "count": 5,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9055.957323501078,
      "r2": 0.8987266414319106
    },
    {
      "method": "LassoCV",
      "count": 1,
      "features": [
        "R&D Spend"
      ],
      "rmse": 7714.334890830185,
      "r2": 0.9265108109341951
    },
    {
      "method": "LassoCV",
      "count": 2,
      "features": [
        "R&D Spend",
        "Marketing Spend"
      ],
      "rmse": 8206.32881316585,
      "r2": 0.9168381183550247
    },
    {
      "method": "LassoCV",
      "count": 3,
      "features": [
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 8995.90580336142,
      "r2": 0.900065308303732
    },
    {
      "method": "LassoCV",
      "count": 4,
      "features": [
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9010.31225499439,
      "r2": 0.899744972111299
    },
    {
      "method": "LassoCV",
      "count": 5,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9055.957323501078,
      "r2": 0.8987266414319106
    },
    {
      "method": "Backward",
      "count": 1,
      "features": [
        "R&D Spend"
      ],
      "rmse": 7714.334890830185,
      "r2": 0.9265108109341951
    },
    {
      "method": "Backward",
      "count": 2,
      "features": [
        "R&D Spend",
        "Marketing Spend"
      ],
      "rmse": 8206.32881316585,
      "r2": 0.9168381183550247
    },
    {
      "method": "Backward",
      "count": 3,
      "features": [
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 8995.90580336142,
      "r2": 0.900065308303732
    },
    {
      "method": "Backward",
      "count": 4,
      "features": [
        "State_New York",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9015.112541602854,
      "r2": 0.8996381209725023
    },
    {
      "method": "Backward",
      "count": 5,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9055.957323501078,
      "r2": 0.8987266414319106
    },
    {
      "method": "SFS",
      "count": 1,
      "features": [
        "R&D Spend"
      ],
      "rmse": 7714.334890830185,
      "r2": 0.9265108109341951
    },
    {
      "method": "SFS",
      "count": 2,
      "features": [
        "R&D Spend",
        "Administration"
      ],
      "rmse": 9152.274782350905,
      "r2": 0.8965609365509308
    },
    {
      "method": "SFS",
      "count": 3,
      "features": [
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 8995.90580336142,
      "r2": 0.900065308303732
    },
    {
      "method": "SFS",
      "count": 4,
      "features": [
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9010.31225499439,
      "r2": 0.899744972111299
    },
    {
      "method": "SFS",
      "count": 5,
      "features": [
        "State_New York",
        "State_California",
        "R&D Spend",
        "Administration",
        "Marketing Spend"
      ],
      "rmse": 9055.957323501078,
      "r2": 0.8987266414319106
    }
  ],
  "automatic": [
    {
      "method": "Backward",
      "count": 1,
      "cv_rmse": 9744.126258464728,
      "test_rmse": 7714.334890830185,
      "test_r2": 0.9265108109341951
    },
    {
      "method": "LassoCV",
      "count": 2,
      "cv_rmse": 9764.065475571411,
      "test_rmse": 8206.32881316585,
      "test_r2": 0.9168381183550247
    },
    {
      "method": "SFS",
      "count": 2,
      "cv_rmse": 9866.28827350089,
      "test_rmse": 9152.274782350905,
      "test_r2": 0.8965609365509308
    },
    {
      "method": "EFS",
      "count": 2,
      "cv_rmse": 9866.28827350089,
      "test_rmse": 9152.274782350905,
      "test_r2": 0.8965609365509308
    },
    {
      "method": "Baseline",
      "count": 5,
      "cv_rmse": 9994.276912406302,
      "test_rmse": 9055.957323501078,
      "test_r2": 0.8987266414319106
    },
    {
      "method": "RFECV",
      "count": 4,
      "cv_rmse": 10399.096681207397,
      "test_rmse": 9316.536408854245,
      "test_r2": 0.8928146452937172
    }
  ]
};
