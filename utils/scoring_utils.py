def relationship_label(domain, alignment):

    if domain > 70 and alignment > 70:
        return "Strong Strategic Alignment"

    if domain > 70 and alignment < 40:
        return "Competitive or Adversarial Relationship"

    if domain < 40 and alignment > 70:
        return "Shared Values but Different Domains"

    return "Low Strategic Connection"