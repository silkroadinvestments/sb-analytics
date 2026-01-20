"""
Sablemoore Analytics - AI Agent
Autonomous AI agent for litigation finance analysis with intelligent duplicate detection
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher

# Try to import LLM libraries
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class DuplicateMatch:
    """Represents a potential duplicate case match"""
    case_1_id: str
    case_2_id: str
    confidence: str  # 'definite', 'likely', 'possible', 'unlikely'
    similarity_score: float
    reasons: List[str]
    ai_explanation: str
    case_1_summary: Dict
    case_2_summary: Dict


class IntelligentDuplicateDetector:
    """
    AI-powered duplicate detection that understands case context.

    Unlike ML-based detection that relies on numerical similarity scores,
    this uses LLM intelligence to understand:
    - Same parties with different name spellings
    - Same incident described differently
    - Related cases (appeals, counterclaims)
    - Cases that share key facts but are genuinely different
    """

    DUPLICATE_ANALYSIS_PROMPT = """You are an expert legal analyst specializing in identifying duplicate litigation cases.

Analyze the following two cases and determine if they are duplicates or related cases.

CASE 1:
{case1_text}

CASE 2:
{case2_text}

Consider the following factors:
1. **Party Names**: Are the claimants/defendants the same people or companies? Account for:
   - Spelling variations (Smith vs Smyth, John vs Jon)
   - Nicknames (William/Bill, Robert/Bob)
   - Company name variations (Ltd vs Limited, & vs and)
   - Maiden/married names

2. **Facts of the Case**: Do they describe the same incident or dispute?
   - Same dates or time periods
   - Same locations or properties
   - Same underlying events

3. **Legal Claims**: Are the same causes of action being pursued?
   - Same type of claim (contract, negligence, etc.)
   - Same or similar remedies sought
   - Same legal basis

4. **Claim Amounts**: Are the amounts similar or related?
   - Exact match suggests duplicate
   - Related amounts might indicate split claims

5. **Jurisdictions**: Same or related courts?

Respond in the following JSON format:
{{
    "is_duplicate": true/false,
    "confidence": "definite|likely|possible|unlikely",
    "similarity_score": 0-100,
    "matching_factors": [
        {{"factor": "Party Names", "match": true/false, "details": "explanation"}},
        {{"factor": "Facts", "match": true/false, "details": "explanation"}},
        {{"factor": "Legal Claims", "match": true/false, "details": "explanation"}},
        {{"factor": "Amounts", "match": true/false, "details": "explanation"}},
        {{"factor": "Jurisdiction", "match": true/false, "details": "explanation"}}
    ],
    "explanation": "A clear explanation of why these cases are or aren't duplicates",
    "recommendation": "What action to take (merge, review, keep separate, etc.)"
}}"""

    def __init__(self):
        self.provider = None
        self.client = None
        self.model = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client"""
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and anthropic_key:
            self.client = anthropic.Anthropic(api_key=anthropic_key)
            self.provider = 'anthropic'
            self.model = 'claude-sonnet-4-20250514'
            return

        openai_key = os.environ.get('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and openai_key:
            self.client = openai.OpenAI(api_key=openai_key)
            self.provider = 'openai'
            self.model = 'gpt-4-turbo-preview'
            return

        self.provider = 'rule_based'

    def _format_case_for_analysis(self, case: Dict) -> str:
        """Format case data for LLM analysis"""
        parts = []

        if case.get('case_id'):
            parts.append(f"Case ID: {case['case_id']}")
        if case.get('case_type'):
            parts.append(f"Case Type: {case['case_type']}")
        if case.get('jurisdiction'):
            parts.append(f"Jurisdiction: {case['jurisdiction']}")
        if case.get('claim_amount'):
            parts.append(f"Claim Amount: £{case['claim_amount']:,.2f}")
        if case.get('defendant_type'):
            parts.append(f"Defendant Type: {case['defendant_type']}")
        if case.get('complexity'):
            parts.append(f"Complexity: {case['complexity']}")
        if case.get('estimated_duration_months'):
            parts.append(f"Est. Duration: {case['estimated_duration_months']} months")
        if case.get('raw_text'):
            # Include first 2000 chars of raw text
            text = case['raw_text'][:2000]
            parts.append(f"\nCase Text:\n{text}")

        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return response.choices[0].message.content
        else:
            return None

    def _extract_names_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract party names from case text"""
        defendants = []
        claimants = []

        if not text:
            return {'defendants': defendants, 'claimants': claimants}

        # Patterns for extracting names
        claimant_patterns = [
            r'claimant[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'plaintiff[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+(?:v\.?|vs\.?|versus)\s+',
        ]

        defendant_patterns = [
            r'defendant[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'respondent[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'(?:v\.?|vs\.?|versus)\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
        ]

        for pattern in claimant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claimants.extend(matches)

        for pattern in defendant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            defendants.extend(matches)

        # Deduplicate
        return {
            'defendants': list(set(defendants)),
            'claimants': list(set(claimants))
        }

    def _quick_similarity_check(self, case1: Dict, case2: Dict) -> Tuple[float, List[str]]:
        """Quick rule-based similarity check for pre-filtering"""
        score = 0.0
        reasons = []

        # Same case type
        if case1.get('case_type') == case2.get('case_type'):
            score += 0.1
            reasons.append("Same case type")

        # Same jurisdiction
        if case1.get('jurisdiction') == case2.get('jurisdiction'):
            score += 0.05
            reasons.append("Same jurisdiction")

        # Similar amount (within 10%)
        amt1 = case1.get('claim_amount', 0)
        amt2 = case2.get('claim_amount', 0)
        if amt1 > 0 and amt2 > 0:
            ratio = min(amt1, amt2) / max(amt1, amt2)
            if ratio > 0.9:
                score += 0.15
                reasons.append(f"Very similar amounts ({ratio*100:.0f}% match)")
            elif ratio > 0.7:
                score += 0.08
                reasons.append(f"Similar amounts ({ratio*100:.0f}% match)")

        # Text similarity
        text1 = case1.get('raw_text', '')[:1000].lower()
        text2 = case2.get('raw_text', '')[:1000].lower()
        if text1 and text2:
            text_sim = SequenceMatcher(None, text1, text2).ratio()
            if text_sim > 0.6:
                score += 0.4
                reasons.append(f"High text similarity ({text_sim*100:.0f}%)")
            elif text_sim > 0.4:
                score += 0.2
                reasons.append(f"Moderate text similarity ({text_sim*100:.0f}%)")
            elif text_sim > 0.25:
                score += 0.1
                reasons.append(f"Some text similarity ({text_sim*100:.0f}%)")

        # Name matching
        names1 = self._extract_names_from_text(case1.get('raw_text', ''))
        names2 = self._extract_names_from_text(case2.get('raw_text', ''))

        # Check defendant overlap
        for d1 in names1['defendants']:
            for d2 in names2['defendants']:
                if self._names_match(d1, d2):
                    score += 0.25
                    reasons.append(f"Matching defendant: {d1} ~ {d2}")
                    break

        # Check claimant overlap
        for c1 in names1['claimants']:
            for c2 in names2['claimants']:
                if self._names_match(c1, c2):
                    score += 0.25
                    reasons.append(f"Matching claimant: {c1} ~ {c2}")
                    break

        return min(score, 1.0), reasons

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy)"""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Exact match
        if n1 == n2:
            return True

        # High similarity
        if SequenceMatcher(None, n1, n2).ratio() > 0.8:
            return True

        # Check if one contains the other
        if n1 in n2 or n2 in n1:
            return True

        # Check common variations
        variations = {
            'william': ['bill', 'will', 'billy'],
            'robert': ['bob', 'rob', 'bobby'],
            'richard': ['rick', 'dick', 'rich'],
            'john': ['jon', 'johnny', 'jack'],
            'james': ['jim', 'jimmy', 'jamie'],
            'michael': ['mike', 'mick', 'mikey'],
            'limited': ['ltd'],
            'company': ['co'],
            'incorporated': ['inc'],
            'and': ['&'],
        }

        for base, alts in variations.items():
            if base in n1 or any(alt in n1 for alt in alts):
                if base in n2 or any(alt in n2 for alt in alts):
                    return True

        return False

    def _rule_based_analysis(self, case1: Dict, case2: Dict) -> Dict:
        """Perform rule-based duplicate analysis when no LLM available"""
        score, reasons = self._quick_similarity_check(case1, case2)

        # Determine confidence
        if score >= 0.8:
            confidence = 'definite'
            is_duplicate = True
        elif score >= 0.6:
            confidence = 'likely'
            is_duplicate = True
        elif score >= 0.4:
            confidence = 'possible'
            is_duplicate = True
        else:
            confidence = 'unlikely'
            is_duplicate = False

        return {
            'is_duplicate': is_duplicate,
            'confidence': confidence,
            'similarity_score': round(score * 100, 1),
            'matching_factors': [{'factor': r, 'match': True, 'details': ''} for r in reasons],
            'explanation': f"Based on rule-based analysis: {', '.join(reasons) if reasons else 'No strong matching factors found'}",
            'recommendation': 'Review manually' if is_duplicate else 'Likely separate cases'
        }

    def analyze_pair(self, case1: Dict, case2: Dict) -> Dict:
        """
        Analyze a pair of cases for potential duplication using AI.
        Returns detailed analysis with confidence and explanation.
        """
        # Format cases for analysis
        case1_text = self._format_case_for_analysis(case1)
        case2_text = self._format_case_for_analysis(case2)

        # If LLM available, use it
        if self.provider != 'rule_based':
            prompt = self.DUPLICATE_ANALYSIS_PROMPT.format(
                case1_text=case1_text,
                case2_text=case2_text
            )

            try:
                response = self._call_llm(prompt)

                # Parse JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    result['source'] = 'ai'
                    result['model'] = self.model
                    return result
            except Exception as e:
                print(f"[AI Duplicate Detector] LLM error: {e}")

        # Fallback to rule-based
        result = self._rule_based_analysis(case1, case2)
        result['source'] = 'rule_based'
        return result

    def find_duplicates(self, cases: List[Dict], min_confidence: str = 'possible') -> List[Dict]:
        """
        Find all potential duplicates in a list of cases.

        Args:
            cases: List of case dictionaries
            min_confidence: Minimum confidence level to include ('definite', 'likely', 'possible', 'unlikely')

        Returns:
            List of duplicate matches with AI analysis
        """
        if len(cases) < 2:
            return []

        confidence_levels = ['unlikely', 'possible', 'likely', 'definite']
        min_level_idx = confidence_levels.index(min_confidence)

        duplicates = []

        # First pass: quick filter using rule-based similarity
        candidates = []
        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                score, reasons = self._quick_similarity_check(cases[i], cases[j])
                if score >= 0.25:  # Pre-filter threshold
                    candidates.append((i, j, score, reasons))

        # Sort by score (highest first) and limit to top candidates for AI analysis
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:50]  # Limit AI calls

        # Second pass: detailed AI analysis on candidates
        for i, j, quick_score, quick_reasons in candidates:
            analysis = self.analyze_pair(cases[i], cases[j])

            confidence = analysis.get('confidence', 'unlikely')
            confidence_idx = confidence_levels.index(confidence) if confidence in confidence_levels else 0

            if confidence_idx >= min_level_idx:
                duplicates.append({
                    'case_1': cases[i]['case_id'],
                    'case_2': cases[j]['case_id'],
                    'case_1_details': cases[i],
                    'case_2_details': cases[j],
                    'similarity': analysis.get('similarity_score', quick_score * 100),
                    'confidence': confidence,
                    'is_duplicate': analysis.get('is_duplicate', False),
                    'matching_factors': analysis.get('matching_factors', []),
                    'explanation': analysis.get('explanation', ''),
                    'recommendation': analysis.get('recommendation', ''),
                    'source': analysis.get('source', 'unknown'),
                    'features': {
                        'text_similarity': quick_score * 100,
                        'quick_reasons': quick_reasons
                    }
                })

        # Sort by similarity score
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def get_training_stats(self) -> Dict:
        """Get stats for compatibility with existing interface"""
        return {
            'total_examples': 0,
            'duplicates': 0,
            'non_duplicates': 0,
            'is_trained': self.provider != 'rule_based',
            'provider': self.provider,
            'model': self.model
        }


class LitigationAIAgent:
    """
    Autonomous AI Agent for Litigation Finance Analysis

    This agent can:
    - Intelligently detect duplicate cases using AI understanding
    - Analyze cases and provide recommendations
    - Answer questions about the portfolio
    - Execute multi-step analysis workflows
    """

    SYSTEM_PROMPT = """You are an expert AI agent specialized in UK litigation finance analysis.
You work for Sablemoore Analytics, a litigation finance firm that invests in legal cases.

Your capabilities include:
1. **Intelligent Duplicate Detection**: You can identify duplicate or related cases by understanding context, not just matching keywords. You recognize name variations, related incidents, and connected cases.

2. **Case Analysis**: Analyze litigation cases for investment potential, considering UK legal precedents and market factors.

3. **Risk Assessment**: Evaluate risk factors and success probability based on case characteristics.

4. **Portfolio Management**: Provide insights on the overall case portfolio.

When detecting duplicates, you look beyond simple text matching to understand:
- Same parties with different name spellings (Smith/Smyth, Ltd/Limited)
- Same incident described differently in separate filings
- Related cases like appeals or counterclaims
- Cases sharing key facts that may or may not be true duplicates

Always provide clear explanations for your conclusions and actionable recommendations."""

    def __init__(self):
        self.provider = None
        self.client = None
        self.model = None
        self.duplicate_detector = IntelligentDuplicateDetector()
        self.conversation_history: List[Dict] = []
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client"""
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and anthropic_key:
            self.client = anthropic.Anthropic(api_key=anthropic_key)
            self.provider = 'anthropic'
            self.model = 'claude-sonnet-4-20250514'
            print("[AIAgent] Initialized with Anthropic Claude")
            return

        openai_key = os.environ.get('OPENAI_API_KEY')
        if OPENAI_AVAILABLE and openai_key:
            self.client = openai.OpenAI(api_key=openai_key)
            self.provider = 'openai'
            self.model = 'gpt-4-turbo-preview'
            print("[AIAgent] Initialized with OpenAI GPT-4")
            return

        self.provider = 'rule_based'
        print("[AIAgent] No LLM available, using rule-based mode")

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'provider': self.provider,
            'model': self.model,
            'llm_available': self.provider != 'rule_based',
            'duplicate_detector_ready': True,
            'conversation_length': len(self.conversation_history)
        }

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM with messages"""
        if self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=messages
            )
            return response.content[0].text
        elif self.provider == 'openai':
            all_messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                max_tokens=4096
            )
            return response.choices[0].message.content
        else:
            return self._rule_based_response(messages[-1]['content'] if messages else "")

    def _rule_based_response(self, query: str) -> str:
        """Generate rule-based response when no LLM available"""
        query_lower = query.lower()

        if 'duplicate' in query_lower or 'similar' in query_lower:
            return """I can help you find duplicate cases. I use intelligent analysis to detect:

1. **Same parties with name variations** - Smith vs Smyth, Ltd vs Limited
2. **Same incident** - Cases describing the same event differently
3. **Related cases** - Appeals, counterclaims, connected matters
4. **Matching key facts** - Similar amounts, dates, locations

To scan for duplicates, use the "AI Duplicate Scanner" feature. I'll analyze each potential match and explain why I think they might be duplicates.

Would you like me to scan your cases now?"""

        elif 'analyze' in query_lower or 'case' in query_lower:
            return """I can provide detailed analysis of any case in your portfolio. For each case, I evaluate:

- **Success probability** based on UK litigation patterns
- **Risk factors** that could affect the outcome
- **Investment recommendation** with supporting rationale
- **Similar cases** in your portfolio

Which case would you like me to analyze? Please provide the case ID."""

        elif 'portfolio' in query_lower or 'overview' in query_lower:
            return """I can provide insights on your entire portfolio including:

- Overall risk distribution
- Concentration by case type and jurisdiction
- Expected returns analysis
- Potential duplicate exposure
- Recommendations for portfolio optimization

Would you like a full portfolio analysis?"""

        else:
            return """I'm your AI litigation finance agent. I specialize in:

1. **Intelligent Duplicate Detection** - Finding related cases that ML might miss
2. **Case Analysis** - Deep analysis of individual cases
3. **Portfolio Insights** - Overview and recommendations for your portfolio
4. **Risk Assessment** - Evaluating success probability and risk factors

What would you like me to help you with?"""

    def chat(self, message: str, cases: List[Dict] = None, predictions: List[Dict] = None) -> Dict:
        """
        Have a conversation with the agent.

        The agent can access cases and predictions to provide informed responses.
        """
        # Add message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Check for duplicate-related queries
        message_lower = message.lower()

        if cases and ('duplicate' in message_lower or 'scan' in message_lower or 'similar' in message_lower):
            # Perform duplicate detection
            duplicates = self.duplicate_detector.find_duplicates(cases, min_confidence='possible')

            if duplicates:
                response = f"I found **{len(duplicates)} potential duplicate(s)** in your portfolio:\n\n"
                for i, dup in enumerate(duplicates[:5], 1):
                    response += f"### Match {i}: Cases #{dup['case_1']} and #{dup['case_2']}\n"
                    response += f"- **Confidence**: {dup['confidence'].upper()}\n"
                    response += f"- **Similarity**: {dup['similarity']:.1f}%\n"
                    response += f"- **Explanation**: {dup['explanation']}\n"
                    response += f"- **Recommendation**: {dup['recommendation']}\n\n"

                if len(duplicates) > 5:
                    response += f"\n*...and {len(duplicates) - 5} more potential matches.*\n"

                response += "\nWould you like me to analyze any of these matches in more detail?"
            else:
                response = "I scanned your portfolio and didn't find any potential duplicates. All cases appear to be unique."

        elif cases and ('portfolio' in message_lower or 'overview' in message_lower or 'summary' in message_lower):
            # Generate portfolio summary
            total = len(cases)
            total_exposure = sum(c.get('claim_amount', 0) for c in cases)

            types = {}
            for c in cases:
                t = c.get('case_type', 'Unknown')
                types[t] = types.get(t, 0) + 1

            response = f"## Portfolio Summary\n\n"
            response += f"- **Total Cases**: {total}\n"
            response += f"- **Total Exposure**: £{total_exposure:,.0f}\n\n"
            response += "### Case Types:\n"
            for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                response += f"- {t}: {count} cases\n"

            if predictions:
                avg_success = sum(p.get('success_rate', 0) for p in predictions) / len(predictions)
                high_risk = sum(1 for p in predictions if p.get('risk_level') == 'High')
                response += f"\n### Risk Analysis:\n"
                response += f"- **Average Success Rate**: {avg_success:.1f}%\n"
                response += f"- **High Risk Cases**: {high_risk}\n"

        else:
            # General conversation
            if self.provider != 'rule_based':
                # Build context with case info
                context = message
                if cases:
                    context += f"\n\n[Context: The portfolio contains {len(cases)} cases with total exposure of £{sum(c.get('claim_amount', 0) for c in cases):,.0f}]"

                self.conversation_history[-1]['content'] = context
                response = self._call_llm(self.conversation_history)
            else:
                response = self._rule_based_response(message)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return {
            'response': response,
            'source': self.provider,
            'model': self.model
        }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def find_duplicates(self, cases: List[Dict], min_confidence: str = 'possible') -> List[Dict]:
        """Find duplicates using intelligent detection"""
        return self.duplicate_detector.find_duplicates(cases, min_confidence)

    def analyze_duplicate_pair(self, case1: Dict, case2: Dict) -> Dict:
        """Analyze a specific pair of cases for duplication"""
        return self.duplicate_detector.analyze_pair(case1, case2)


# Singleton instance
_agent = None
_duplicate_detector = None


def get_ai_agent() -> LitigationAIAgent:
    """Get singleton AI agent instance"""
    global _agent
    if _agent is None:
        _agent = LitigationAIAgent()
    return _agent


def get_intelligent_duplicate_detector() -> IntelligentDuplicateDetector:
    """Get singleton intelligent duplicate detector"""
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = IntelligentDuplicateDetector()
    return _duplicate_detector
