vector < int > find_common_nums(vector < int > first, vector < int > second){
	unordered_map < int , int > mymap;
	unordered_map < int , int >::iterator it;
	for(int i = 0; i < first.size; i++){
		mymap.insert(std::make_pair(first[i], i));
	}
	vector < int > ans;
	for(int i = 0; i < second.size() && i < first.size(); i++){
		it = mymap.find(second[i]);
		if(it != mymap.end()){
			ans.push_back(second[i]);
		}
	}
	return ans; 
}