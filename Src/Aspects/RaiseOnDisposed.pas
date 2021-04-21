namespace TensorFlow.Island.Aspects;

interface

uses
  RemObjects.Elements.Cirrus,
  RemObjects.Elements.Cirrus.Statements,
  RemObjects.Elements.Cirrus.Values;

type
  [AttributeUsage(AttributeTargets.Class)]
  RaiseOnDisposedAttribute = public class(Attribute, IMethodImplementationDecorator)
  public
    method HandleImplementation(Services: IServices; aMethod: IMethodDefinition);
  end;

implementation

method RaiseOnDIsposedAttribute.HandleImplementation(Services: IServices; aMethod: IMethodDefinition);
begin
  var lExitCondition := 
     String.Equals(aMethod.Name, '.', StringComparison.OrdinalIgnoreCase) or
     String.Equals(aMethod.Name, '~', StringComparison.OrdinalIgnoreCase) or
     (aMethod.Visibility <> Visibility.Public) or 
     aMethod.Static;
  
  if lExitCondition then
    exit;

  var lStandaloneStatement := new StandaloneStatement(new ProcValue(new SelfValue, 'CheckAndRaiseOnDisposed'));
  var lPlaceHolderStatement := new PlaceHolderStatement;
  var lBeginStatement := new BeginStatement(lStandaloneStatement, lPlaceHolderStatement);
  aMethod.ReplaceMethodBody(lBeginStatement);
end;

end.