from typing import List, Dict, Any
from datetime import datetime

def rank_search_results(findings: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Nadaje ranking wynikom wyszukiwania.
    """
    for finding in findings:
        # Bazowy ranking od jakości dopasowania
        base_rank = finding.get('match_quality', 0.5)
        
        # Bonus za dokładne dopasowanie
        if finding.get('is_exact_match', False):
            base_rank += 0.3
            
        # Bonus za pozycję w dokumencie 
        position_score = 0
        if 'position' in finding:
            # Podstawowy bonus za wcześniejszą pozycję
            position = finding.get('position', 0)
            position_score = max(0, 0.1)  # Stały bonus
                
        finding['rank'] = min(1.0, base_rank + position_score)
    
    # Sortowanie od najwyższego rankingu
    ranked_findings = sorted(findings, key=lambda x: x.get('rank', 0), reverse=True)
    
    return ranked_findingss