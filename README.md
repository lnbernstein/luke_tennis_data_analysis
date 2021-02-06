# luke_tennis_data_analysis
All Data is copyright Jeff Sackmann under Creative Commons Attribution-NonCommerical-ShareAlike 4.0 international License

https://github.com/JeffSackmann/tennis_atp

Scenario 3: Attempts to decipher which player won the match
<ul>
    <li> data_clean method filters the data into unique rows for winner and loser, assigns 1 and 0 for win and loss respectively, concats two data frames together and drops all rows with null values</li>
    <li> select_features uses Scikit Learn's Select K Best and Extra Trees Classifer to determine which features are most effective for use in models</li>
    <li>Features selected were 'bpFaced', '1stWon', 'ace', 'bpSaved', '2ndWon' which are the top five from K Best </li>
    <li>Three models implemented: Logistic Regression, Decision Tree, and Random Forest</li>
    <li>Best Precision is found in logistic Regression with .78 on average</li>
    <li>Best Area under ROC however is found in Decision Tree with .85 on average</li>
</ul>

Scenario 1 finds the ten players with the most wins and losses since 1985

break points computes percentages for the ten players with the most wins and losses for break points saved and break points converted
next steps are to find ace percentages, and see if I can perdict which player will win the match based on break point conversion and ace percentage

MIT License

Copyright (c) 2021 Luke Bernstein 